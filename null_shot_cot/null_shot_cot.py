from pathlib import Path

from llm4pcg.competition import chat_with_llm, run_evaluation
from llm4pcg.models.trial_context import TrialContext
from llm4pcg.models.trial_loop import TrialLoop

class NullShotChainOfThoughtPrompting(TrialLoop):
    @staticmethod
    def run(ctx: TrialContext, target_character: str) -> str:
        """
        Runs the null-shot chain-of-thought prompting.
        :param ctx: The trial context.
        :param target_character: The target character.
        :return: The generated text.
        """
        prompt_template = open(Path("prompts/null_shot_cot.txt"), "r").read()
        response = chat_with_llm(ctx, [{"role": "user", "content": prompt_template
                                     .replace("<OBJECT>", target_character)}])
        response = response[0]
        return response

if __name__ == "__main__":
    model_name = '<MODEL_NAME>'
    # local_model_base_url = e.g., 'http://localhost:1313/v1', 'http://localhost:11434/v1'
    local_model_base_url = '<LOCAL_MODEL_BASE_URL>'
    # team_name = '<TEAM_NAME>'
    team_name = 'null_shot_cot'
    run_evaluation(team_name=team_name, fn=NullShotChainOfThoughtPrompting, 
                    model_name=model_name,
                    local_model_base_url=local_model_base_url,
                    num_trials=1, 
                    characters=["A", "B", "C"])

