FROM registry.access.redhat.com/ubi9/python-312

WORKDIR /opt/app-root/src

COPY pyproject.toml README.md ./
COPY src ./src
COPY docs/api/openapi.yaml ./docs/api/openapi.yaml

# Install the exact RHOAI 3.6 Ray wheel tested with the Ray 2.55.1 runtime.
# Both supported Linux architectures are available in the product index.
RUN case "$(uname -m)" in \
      x86_64) wheel_arch=x86_64 ;; \
      aarch64) wheel_arch=aarch64 ;; \
      *) echo "Unsupported architecture for RHOAI Ray wheel" >&2; exit 1 ;; \
    esac && \
    pip install --no-cache-dir \
      "ray[data,default] @ https://packages.redhat.com/api/pulp-content/public-rhai/rhoai/3.6/cpu-ubi9/ray-2.55.1-4-cp312-cp312-linux_${wheel_arch}.whl" \
      "colorama @ https://packages.redhat.com/api/pulp-content/public-rhai/rhoai/3.6/cpu-ubi9/colorama-0.4.6-1-py2.py3-none-any.whl" && \
    pip install --no-cache-dir .

USER 1001
EXPOSE 8080

CMD ["uvicorn", "training_service.main:app", "--host", "0.0.0.0", "--port", "8080"]
