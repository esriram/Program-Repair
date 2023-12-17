#!/bin/bash

function install_requirements() {
    pip3 install tqdm==4.64.0;
    pip3 install nltk==3.6.7;
    pip3 install wordninja==2.0.0;
    pip3 install torch torchaudio torchvision;
    pip3 install datasets==1.18.3;
    pip3 install transformers==4.16.2;
}

install_requirements;
