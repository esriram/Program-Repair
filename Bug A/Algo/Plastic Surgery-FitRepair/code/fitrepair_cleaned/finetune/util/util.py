import random
from timeit import default_timer as get_now

import numpy as np
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ConstantLR, LambdaLR
from util.input_args import ScheduleArgs


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Model/Dataset related
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def get_optimizer(type, lr, betas, eps, weight_decay, params):
    if type == "adamw":
        return AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif type == "adam":
        return Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    else:
        raise NotImplementedError


class LinearCurve:
    def get_warmup(self, value):
        return value

    def get_decay(self, total, current, total_warmup):
        return max(0.0, total - current) / max(1.0, total - total_warmup)


class ExpCurve:
    def __init__(self, schedule_args):
        self.schedule_args = schedule_args

    def get_warmup(self, value):
        return value**2

    def get_decay(self, total, current, total_warmup):
        return self.schedule_args.decay_rate ** (
            (current - total_warmup) / self.schedule_args.decay_step
        )


class BaseScheduler:
    def __init__(self, schedule_args):
        self.schedule_args = schedule_args

    def still_in_warmup(self, position):
        return position < self.get_total_warmup()

    def get_total_warmup(self):
        return self.get_total() * self.schedule_args.warmup_proportion

    def get_warmup_percent(self, position):
        return position / self.get_total_warmup()


class StepScheduler(BaseScheduler):
    def __init__(self, schedule_args, extra_args):
        super().__init__(schedule_args)
        self.extra_args = extra_args

    def get_correct_position(self, current_step):
        return current_step

    def get_total(self):
        return self.extra_args.max_steps


class FixedWarmupScheduler(StepScheduler):
    def __init__(self, schedule_args, extra_args):
        super().__init__(schedule_args, extra_args)

    def get_total_warmup(self):
        return self.schedule_args.num_warmup_steps


class TimeScheduler(BaseScheduler):
    def __init__(self, schedule_args, extra_args):
        super().__init__(schedule_args)
        self.extra_args = extra_args

    def get_correct_position(self, current_step):
        return (get_now() - self.extra_args.exp_start_marker) / 3600

    def get_total(self):
        return self.extra_args.total_training_time


CURVES = {"linear": lambda args: LinearCurve(), "exp": lambda args: ExpCurve(args)}
SCHEDULES = {
    "step": StepScheduler,
    "constant_step": FixedWarmupScheduler,
    "time": TimeScheduler,
}


def get_scheduler(schedule_args: ScheduleArgs, optimizer, extra_args):
    if schedule_args.curve == "constant":
        return ConstantLR(optimizer, factor=1, total_iters=0)
    curver = CURVES[schedule_args.curve](schedule_args)
    scheduler = SCHEDULES[schedule_args.lr_schedule](schedule_args, extra_args)
    return build_scheduler(optimizer, scheduler, curver)


def build_scheduler(optimizer, scheduler, curver):
    def get_warmup_calc(current_step: int):

        position = scheduler.get_correct_position(current_step)
        # _file = open('warmup.txt', mode='a')
        # _file.write(str(position))
        # _file.close()

        if scheduler.still_in_warmup(position):
            warmup_position = scheduler.get_warmup_percent(position)
            return curver.get_warmup(warmup_position)
        else:
            return curver.get_decay(
                scheduler.get_total(), position, scheduler.get_total_warmup()
            )

    return LambdaLR(optimizer, get_warmup_calc, last_epoch=-1)


def check_token_length(tokenizer, code, max_length=512):
    if len(tokenizer(code)["input_ids"]) <= max_length:
        return True
    else:
        return False


def get_index(lst=None, item=""):
    return [index for (index, value) in enumerate(lst) if value == item]


def merge_intervals(intervals):
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] + 1 < interval[0]:
            merged.append(interval)
        else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged


def pad_to_len(pair_targets, pad, max_pair_target_len):
    for i in range(len(pair_targets)):
        pair_targets[i] = pair_targets[i][:max_pair_target_len]
        this_len = len(pair_targets[i])
        for j in range(max_pair_target_len - this_len):
            pair_targets[i].append(pad)
    return pair_targets


def collate_2d(values, pad_idx, left_pad, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size_0 = max(v.size(0) for v in values)
    size_1 = max(v.size(1) for v in values)
    res = values[0].new(len(values), size_0, size_1).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(
            v,
            res[i, size_0 - v.size(0) :, size_1 - v.size(1) :]
            if left_pad
            else res[i, : v.size(0), : v.size(1)],
        )
    return res


def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]


def handle_special_cases_for_mask_code_lines(code_line_without_space, language, idx):
    if language == "php":
        if (
            code_line_without_space
            == "publicfunctiontruncate($string,$length=80,$etc='…',$break_words=false,$middle=false)"
        ):
            return "publicfunctiontruncate($string,$length=80,$etc='…',$reak_words=flse,$iddle=flse)"
        if (
            code_line_without_space
            == "$this->file_dst_name_body=strtr($this->file_dst_name_body,array('Þ'=>'TH','þ'=>'th','Ð'=>'DH','ð'=>'dh','ß'=>'ss','Œ'=>'OE','œ'=>'oe','Æ'=>'AE','æ'=>'ae','µ'=>'u'));"
        ):
            return "$this->file_dst_name_body=strtr($this->file_dst_name_body,array('Þ'>TH',þ'='h',''=>'D','ð=>dh,'ß'=>'s''Œ'>'O','œ'='oe,Æ'=>'AE'''=>ae','µ=>''));"
        if (
            "publicfunctionpriceEditor($method,$tid,$pid,$group,$prefix="
            in code_line_without_space
        ):
            return code_line_without_space.replace('$suffix=""){', '$sffix="{')
    if language == "java":
        if code_line_without_space == "name=name.replace('/','\\\\');":
            return "name=name.replace('','');"
        if code_line_without_space == "name=name.replace('\\\\','/');":
            return "name=name.replace('','');"
        if code_line_without_space == "if((LA7_0=='d'||LA7_0=='f'||LA7_0=='m')){":
            return "if((LA7_0==''||LA7_0==''||LA7_0=='')){"
        if code_line_without_space == "temp=temp.replace('\\\\','_');":
            return "temp=temp.replace('','');"
        if (
            code_line_without_space
            == "sb.append(key).append('=').append(val).append('\n');"
        ):
            return "sb.append(key).append('').append(val).append('');"
        if (
            code_line_without_space
            == "if(grandParentStr.charAt(0)=='N'&&(parentStr.charAt(0)=='P'||parentStr.charAt(0)=='A')){"
        ):
            return "if(grandParentStr.charAt(0)==''&&(parentStr.charAt(0)==''||parentStr.charAt(0)=='')){"
    if language == "javascript":
        if (
            "if(!file||" in code_line_without_space
            and "file===local.options.namespace){" in code_line_without_space
        ):
            return "if(!file||ile==ocal.options.namespace)"
        if (
            "}elseif(typeof(res)==='string'||typeof(res)==='number'"
            in code_line_without_space
            and "||typeof(res)==='boolean'){" in code_line_without_space
        ):
            return "}elseif(typeof(res)==='string'||typeof(res)==='number'|ypeof(res)==boolean')"
        if (
            "require('utils/merge'):require('../../../../utils/lib/merge');"
            in code_line_without_space
            and "varmerge=(isGFFCtx)?" in code_line_without_space
        ):
            return "varmerge=(isGFFCtx)?equire('utils/merge')equire('../../../../utils/lib/merge');"
        if (
            code_line_without_space
            == "if(!checked||checked=='null'||checked=='false'||checked==''){"
        ):
            return "if(!checked||checked=='null'||hecked=false'|hecked='"
        if (
            code_line_without_space
            == 'token_error(S.token,"Unexpectedtoken"+S.token.type+"«"+S.token.value+"»"+",expected"+type+"«"+val+"»");'
        ):
            return 'token_error(S.token,"Unexpectedtoken"+S.token.type+"«".token.value»"+"expected"+tpe+"«"+a+»);'
        if (
            code_line_without_space
            == 'functionprint(input,opts={},/*…Internal:*/name="",refs=null){'
        ):
            return 'functionprint(input,opts={},/*…Internal:*/nme=",rfs=nll){'
        if (
            code_line_without_space
            == "letmeanString=`Mean:${numbersWithCommas(mean.toFixed(2))}${suffix}`;"
        ):
            return "letmeanString=`${numbersWithCommas(mean.toFixed(2))}${suffix}`;"
    if language == "ruby":
        if code_line_without_space == "defcreate_server(name,region=:'eu-central')":
            return "defcreate_server(name,region=:'')"
        if code_line_without_space == "deflog_instances(key=key_name,status=/running/)":
            return "deflog_instances(key=key_name,status=//)"
        if code_line_without_space == "defhomothétiec1,c2,c3=1.0":
            return "defhomothétie1,2,3=1.0"
        if (
            "defquotes(input,type='\"" in code_line_without_space
            and "start_quotes=" in code_line_without_space
            and ",end_quotes=" in code_line_without_space
        ):
            return code_line_without_space.replace(",e", "")
    if language == "python":
        if code_line_without_space == "def_check_params(ρ,χ,γ,σ,bounds,n):":
            return "deflog_instances(key=key_name,status=//)"
        if (
            "lower_bound=0,upper_bound=100,enable_boundary_glyphs=False):"
            in code_line_without_space
        ):
            return code_line_without_space.replace(
                "upper_bound=100,enable_boundary_glyphs",
                "pper_bound=100enable_boundaryglyphs",
            )
        if (
            code_line_without_space
            == "defupdate(self,τ:float=1.0,update_indicators=True,dampen=False):"
        ):
            return "defupdate(self,τ:loat.0,pdate_indicators=True,ampen=False):"
        if code_line_without_space == "label=str(mini)+'$M_{\odot}$,Z='+str(zini)":
            return "label=str(mini)+'odot}$,Z='+str(zini)"
        if (
            code_line_without_space
            == 'defline(separator="-·-",color=None,padding=None,num=1):'
        ):
            return 'defline(separator="-·-",olor=None,adding=None,um=1):'
        if code_line_without_space == "defΩ(self,k,n,τ,σ):":
            return "defΩ(self,,,,σ:"
    return code_line_without_space
