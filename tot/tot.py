from pathlib import Path

from llm4pcg.competition import chat_with_llm, run_evaluation
from llm4pcg.models.trial_context import TrialContext
from llm4pcg.models.trial_loop import TrialLoop

class TreeOfThoughtPrompting(TrialLoop):
    @staticmethod
    def extract_scores(scores_str: str):
        import re
        scores_str = scores_str.lower()
        stability_pattern = r".*stability: (10|\d).*"
        similarity_pattern = r".*similarity: (10|\d).*"
        stability = 0
        similarity = 0
        if stability_match := re.search(stability_pattern, scores_str):
            stability = int(stability_match.group(1))
        if similarity_match := re.search(similarity_pattern, scores_str):
            similarity = int(similarity_match.group(1))
        return stability, similarity

    @staticmethod
    def tot(ctx: TrialContext, target_character: str):
        max_depth = 2
        branching_factor = 2
        current_content = ""
        try:
            for _ in range(max_depth):
                task_prompt = open(Path("prompts/task.txt"), "r").read().replace("<OBJECT>", target_character).replace(
                    "<GENERATED_CONTENT_SO_FAR>", "Nothing." if current_content == "" else current_content)
                samples = chat_with_llm(ctx, [{
                    "role": "user",
                    "content": task_prompt
                }], n=branching_factor)

                values = []
                for sample in samples:
                    evaluation_prompt = open(Path("prompts/evaluate.txt"), "r").read().replace("<OBJECT>",
                                                                                               target_character).replace(
                        "<GENERATED_CONTENT_SO_FAR>", current_content + "\n" + sample)
                    evaluation_result = chat_with_llm(ctx, [{
                        "role": "user",
                        "content": evaluation_prompt
                    }])
                    scores = TreeOfThoughtPrompting.extract_scores(evaluation_result[0])
                    values.append((sample, scores))
                values = sorted(values, key=lambda x: x[1], reverse=True)
                if current_content == "":
                    current_content = values[0][0]
                else:
                    current_content += "\n" + values[0][0]
            format_prompt = open(Path("prompts/format.txt"), "r").read()
            response = chat_with_llm(ctx, [{
                "role": "user",
                "content": format_prompt.replace("<GENERATED_CONTENT>", current_content)
            }])
            final_response = response[0]
        except (ValueError, TimeoutError) as e:
            print(e)
            final_response = current_content

        return final_response

    @staticmethod
    def run(ctx: TrialContext, target_character: str) -> str:
        """
        Runs the tree-of-thought prompting.
        :param ctx: The trial context.
        :param target_character: The target character.
        :return: The generated text.
        """
        final_response = TreeOfThoughtPrompting.tot(ctx, target_character)

        return final_response

if __name__ == "__main__":
    model_name = '<MODEL_NAME>'
    # local_model_base_url = e.g., 'http://localhost:1313/v1', 'http://localhost:11434/v1'
    local_model_base_url = '<LOCAL_MODEL_BASE_URL>'
    # team_name = '<TEAM_NAME>'
    team_name = 'tree_of_thought'
    run_evaluation(team_name=team_name, fn=TreeOfThoughtPrompting, 
                    model_name=model_name,
                    local_model_base_url=local_model_base_url,
                    num_trials=1, 
                    characters=["A", "B", "C"])
