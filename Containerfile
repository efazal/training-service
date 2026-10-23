FROM registry.access.redhat.com/ubi9/python-312@sha256:f43a57e9a6a2243e333008219002b9f2b1661868a94199f5dec4d602e1501482

WORKDIR /opt/app-root/src

COPY pyproject.toml README.md ./
COPY src ./src
COPY docs/api/openapi.yaml ./docs/api/openapi.yaml

# Install the exact RHOAI 3.6 Ray wheel tested with the Ray 2.55.1 runtime.
# Both supported Linux architectures are available in the product index.
RUN case "$(uname -m)" in \
      x86_64) wheel_arch=x86_64; ray_sha=6f4dcd12e55a9ed2289e207b536554f87f96c21c6bf08f531eebe38cc37e2c72 ;; \
      aarch64) wheel_arch=aarch64; ray_sha=624be1fda4f74bf9bcc9718ea1a4184814a64a79ca748e13b96b3439b6e4b296 ;; \
      *) echo "Unsupported architecture for RHOAI Ray wheel" >&2; exit 1 ;; \
    esac && \
    pip install --no-cache-dir \
      "ray[data,default] @ https://packages.redhat.com/api/pulp-content/public-rhai/rhoai/3.6/cpu-ubi9/ray-2.55.1-4-cp312-cp312-linux_${wheel_arch}.whl#sha256=${ray_sha}" \
      "colorama @ https://packages.redhat.com/api/pulp-content/public-rhai/rhoai/3.6/cpu-ubi9/colorama-0.4.6-1-py2.py3-none-any.whl#sha256=bde866cb771ec23ed0415c915adc359fc57577b7b07821a71f621aed7fc222bb" && \
    pip install --no-cache-dir .

USER 1001
EXPOSE 8080

CMD ["uvicorn", "training_service.main:app", "--host", "0.0.0.0", "--port", "8080"]
