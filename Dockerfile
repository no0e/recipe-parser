FROM python:3.12-slim
WORKDIR /srv
# prebuilt CPU wheel of llama-cpp-python (no compiler needed)
RUN pip install --no-cache-dir llama-cpp-python==0.3.16 \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
 && pip install --no-cache-dir "fastapi>=0.115" "uvicorn>=0.30" "gradio>=5" "pydantic>=2"
ARG MODEL_URL=https://huggingface.co/no0e/recipe-parser-qwen2.5-1.5b-gguf/resolve/main/recipe-parser-q8_0.gguf
ADD ${MODEL_URL} /models/model.gguf
COPY recipe_parser recipe_parser
COPY app app
ENV PORT=8080
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
