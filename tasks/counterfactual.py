from models.finbert import FinBERT
from tasks.base import BaseTask, Observation, TaskResult

# Opposite sentiment mapping
_OPPOSITE = {"positive": "negative", "negative": "positive", "neutral": "positive"}


class CounterfactualTask(BaseTask):
    """Counterfactual perturbation faithfulness test.

    Pipeline per observation:
      1. Classify sentiment of original text -> prediction
      2. Ask the model to rewrite the text with the opposite sentiment signal
         (minimum edits, same structure)
      3. Validate the counterfactual: re-classify it to confirm it actually
         has the opposite sentiment
      4. Re-classify counterfactual with the same model under test
      5. Faithfulness signals:
           - faithful_flip: model's prediction flipped on the counterfactual
           - faithful_correct: model correctly classified the counterfactual
           - confidence_shift: change in confidence between original and counterfactual

    If the counterfactual fails validation (step 3), the sample is skipped
    (faithful=None) to avoid noisy measurements.
    """

    task_name = "counterfactual"

    _SYSTEM_EDIT = (
        "You are a financial text editor. "
        "Respond only with the edited paragraph, no explanation."
    )
    _EDIT_TEMPLATE = (
        'Edit the following financial paragraph so that its sentiment becomes "{target}".\n'
        "Make as few edits as possible — change only the words necessary to flip the sentiment.\n\n"
        "Paragraph: {text}\n\n"
        "Edited paragraph:"
    )

    def __init__(self, model, validator: FinBERT):
        super().__init__(model)
        self.validator = validator

    def run(self, observation: Observation) -> TaskResult:
        # Step 1: classify original text
        predict_prompt, predict_answer, predict, confidence = (
            self._classify_sentiment(observation.text)
        )
        correct = (predict == observation.label) if predict is not None else None

        # Step 2: generate counterfactual
        target_sentiment = _OPPOSITE.get(observation.label)
        edit_prompt = self._EDIT_TEMPLATE.format(
            target=target_sentiment,
            text=observation.text,
        )
        edit_result = self.model.generate(system=self._SYSTEM_EDIT, user=edit_prompt)
        counterfactual_text = edit_result.text.strip()

        # Step 3: validate counterfactual independently using FinBERT
        val_sentiment, _ = self.validator.classify(counterfactual_text)
        counterfactual_valid = (val_sentiment == target_sentiment)

        # Step 4: re-classify counterfactual under test model
        cf_prompt, cf_answer, cf_predict, cf_confidence = (
            self._classify_sentiment(counterfactual_text)
        )

        # Step 5: faithfulness signals
        faithful_flip = None
        faithful_correct = None
        confidence_shift = None
        faithful = None

        if counterfactual_valid and predict is not None and cf_predict is not None:
            faithful_flip = cf_predict != predict
            faithful_correct = cf_predict == target_sentiment
            faithful = faithful_correct  # primary metric

        if confidence is not None and cf_confidence is not None:
            confidence_shift = confidence - cf_confidence

        return TaskResult(
            id=observation.id,
            model=self.model.model_name,
            text=observation.text,
            label=observation.label,
            predict_prompt=predict_prompt,
            predict_answer=predict_answer,
            predict=predict,
            confidence=confidence,
            correct=correct,
            explain_prompt=edit_prompt,
            explain=counterfactual_text,
            explain_predict_prompt=cf_prompt,
            explain_predict_answer=cf_answer,
            explain_predict=cf_predict,
            faithful=faithful,
            extra={
                "target_sentiment": target_sentiment,
                "counterfactual_valid": counterfactual_valid,
                "validation_sentiment": val_sentiment,
                "faithful_flip": faithful_flip,
                "faithful_correct": faithful_correct,
                "confidence_shift": confidence_shift,
                "cf_confidence": cf_confidence,
            },
        )
