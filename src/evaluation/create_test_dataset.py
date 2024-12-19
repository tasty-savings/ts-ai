from langfuse import Langfuse
import openai

langfuse = Langfuse()

langfuse.create_dataset(name="레시피의 맛");