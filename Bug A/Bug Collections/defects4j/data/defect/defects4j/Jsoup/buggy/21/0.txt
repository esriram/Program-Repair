        Or(Collection<Evaluator> evaluators) {
            super();
            if (evaluators.size() > 1)
                this.evaluators.add(new And(evaluators));
            else // 0 or 1
                this.evaluators.addAll(evaluators);
        }