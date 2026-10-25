FROM python:3.12-slim
WORKDIR /srv
# build llama-cpp-python from source with AVX2: the generic prebuilt wheel is ~6x slower
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake && rm -rf /var/lib/apt/lists/* \
 && CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON" \
    pip install --no-cache-dir llama-cpp-python==0.3.16 \
 && pip install --no-cache-dir "fastapi>=0.115" "uvicorn>=0.30" "gradio>=5" "pydantic>=2"
ARG MODEL_URL=https://huggingface.co/no0e/recipe-parser-qwen2.5-1.5b-gguf/resolve/main/recipe-parser-q8_0.gguf
ADD ${MODEL_URL} /models/model.gguf
COPY recipe_parser recipe_parser
COPY app app
ENV PORT=8080 N_THREADS=8
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
