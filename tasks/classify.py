from tasks.base import BaseTask, Observation, TaskResult


class ClassifyTask(BaseTask):
    """Baseline sentiment classification.

    Runs a single inference call per observation and records:
    - the model's predicted sentiment
    - prediction confidence (first-token probability)
    - whether the prediction is correct

    This is not a faithfulness test — it establishes each model's
    accuracy baseline on Financial PhraseBank before any perturbations.
    """

    task_name = "classify"

    def run(self, observation: Observation) -> TaskResult:
        prompt, answer, predict, confidence = self._classify_sentiment(observation.text)
        correct = (predict == observation.label) if predict is not None else None

        return TaskResult(
            id=observation.id,
            model=self.model.model_name,
            text=observation.text,
            label=observation.label,
            predict_prompt=prompt,
            predict_answer=answer,
            predict=predict,
            confidence=confidence,
            correct=correct,
        )
