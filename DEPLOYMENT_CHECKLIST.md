# NID OCR Production Deployment Checklist

Use this checklist for the `codex/t4-ocr-optimizations` branch. Complete every **required** item before sending client traffic to the service.

## Release record

- [ ] Deployment date: `____________________`
- [ ] Deployer: `____________________`
- [ ] Environment: `staging / production`
- [ ] Commit SHA: `____________________`
- [ ] Previous stable commit SHA: `____________________`
- [ ] Rollback owner: `____________________`

## 1. Server and GPU

- [ ] Server has at least 16 vCPU, 32 GB RAM, and 200 GB SSD free-space capacity.
- [ ] GPU is an NVIDIA Tesla T4 with approximately 16 GB VRAM, or a faster compatible NVIDIA GPU.
- [ ] `nvidia-smi` detects the GPU without errors.
- [ ] GPU memory is mostly free before the OCR service starts.
- [ ] NVIDIA driver, CUDA runtime, PyTorch, and Surya versions are mutually compatible.
- [ ] The following command prints `True`, the CUDA version, and the expected GPU name:

  ```bash
  python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"
  ```

- [ ] The server has no other GPU-heavy process competing with this service.
- [ ] Time synchronization, firewall rules, TLS certificate, and the production hostname are configured.

## 2. Clean installation

- [ ] Deploy from the approved commit; do not copy a developer working directory to production.
- [ ] Create a clean Python 3.11 virtual environment.
- [ ] Install the exact package versions proven in staging, including PyTorch with CUDA support, Surya OCR, Pillow, FastAPI, OpenCV, and the packages in `requirements.txt`.
- [ ] Do not use a CPU-only PyTorch build on the T4 server.
- [ ] Confirm imports succeed in the clean environment:

  ```bash
  python -c "import torch, surya, cv2, fastapi; print('dependencies-ok')"
  ```

- [ ] Run the automated checks from the directory containing the `nid_ocr` package:

  ```bash
  python -m unittest discover -s nid_ocr/tests -v
  ```

- [ ] All tests pass with no unexpected warnings or dependency downloads during the test run.

## 3. Production configuration

- [ ] `DEFAULT_OCR_ENGINE=surya`
- [ ] `SURYA_DTYPE=float16`
- [ ] `MAX_LONG_SIDE=2800`
- [ ] `SURYA_RECOGNITION_BATCH_SIZE=64`
- [ ] `SURYA_DETECTOR_BATCH_SIZE=16`
- [ ] `OCR_MAX_CONCURRENCY=2`
- [ ] `OCR_QUEUE_WAIT_SECONDS=5`
- [ ] `OCR_WARMUP_ENABLED=true`
- [ ] `OCR_TIMING_LOG_FILE=logs/ocr_response_times.log`
- [ ] `OCR_TIMING_LOG_MAX_BYTES=10485760`
- [ ] `OCR_TIMING_LOG_BACKUP_COUNT=5`
- [ ] Relative timing-log paths resolve from the `nid_ocr` package directory; confirm that directory is writable by the service account.
- [ ] Environment files are readable only by the service account and are not exposed by the web server.
- [ ] No database credentials or object-storage credentials are required for this stateless deployment.
- [ ] Temporary storage has at least 10 GB free and is monitored for abnormal growth.

## 4. Process and network configuration

- [ ] Run exactly **one Uvicorn worker per T4 GPU**. Multiple workers load duplicate models and can exhaust VRAM.
- [ ] Start from the directory containing the `nid_ocr` package:

  ```bash
  python -m uvicorn nid_ocr.main:app --host 0.0.0.0 --port 8000 --workers 1
  ```

- [ ] The service is managed by systemd, Docker, or another supervisor with automatic restart enabled.
- [ ] The reverse proxy/load balancer sends traffic only after application startup and Surya warm-up complete.
- [ ] Readiness checking uses `/openapi.json` and expects HTTP 200.
- [ ] Proxy request timeout is at least 120 seconds; connection and upload limits match the accepted image size.
- [ ] Only the required API port is reachable; the application port is not publicly exposed when a reverse proxy is used.
- [ ] TLS is enabled for all client requests because NID images contain sensitive data.

## 5. Startup verification

- [ ] Logs show the expected Tesla T4 device and `torch.float16` during Surya initialization.
- [ ] Logs show `Surya OCR models loaded` before the instance receives client traffic.
- [ ] Startup completes without CUDA out-of-memory, CPU fallback, or model-download errors.
- [ ] After warm-up, record GPU usage:
  - GPU memory: `________ MB`
  - GPU utilization while idle: `________ %`
  - Process RAM: `________ GB`
- [ ] `/openapi.json` returns HTTP 200.
- [ ] The API schema contains `/nid/front` and `/nid/back`.
- [ ] The API schema does not expose the old `/uploads` history endpoint.

## 6. Functional and accuracy checks

Use approved, non-production test images with known expected results.

- [ ] Test at least 10 front images: clear, rotated, low contrast, old laminated NID, and smart NID.
- [ ] Test at least 10 back images covering the same quality range.
- [ ] Front responses correctly return name, parent/spouse name, date of birth, NID number, signature, and Bengali-name crop where present.
- [ ] Back responses correctly return address, blood group where present, issue date, and place of birth.
- [ ] All critical values—NID number and date of birth—match the approved ground truth.
- [ ] No accuracy regression is observed compared with the previous production release.
- [ ] Unsupported file types return HTTP 400.
- [ ] Invalid/unreadable images fail safely without crashing the service.
- [ ] Uploaded images, signature crops, and Bengali-name crops are not permanently saved after the response.
- [ ] Application logs do not contain image content, base64 data, NID numbers, or other personal information.
- [ ] `logs/ocr_response_times.log` contains one valid JSON record per OCR request.
- [ ] Each timing record contains `request_id`, `status_code`, `queue_wait_ms`, `processing_ms`, and `total_duration_ms`.
- [ ] The response contains an `X-Request-ID` header matching its timing-log record.

## 7. Performance and load acceptance

Run the same test images and load-test method used for the previous report. Do not approve production from a single manual request.

- [ ] First request is not delayed by model loading because startup warm-up succeeded.
- [ ] Single-user clear-image target:
  - Front median: **35 seconds or less**
  - Back median: **30 seconds or less**
- [ ] Two-concurrent-user test completes without CUDA out-of-memory or application crash.
- [ ] Two-concurrent-user P95 latency is **70 seconds or less**.
- [ ] Successful-request error rate is below **1%**, excluding intentional overload responses.
- [ ] Four-concurrent-user overload test returns controlled HTTP 503 responses with `Retry-After` when both GPU slots remain occupied; requests do not hang indefinitely.
- [ ] GPU memory remains below **14.5 GB** during the peak test, leaving safety margin on a 16 GB T4.
- [ ] CPU remains below **85% sustained**, RAM below **85%**, and the server does not use swap heavily.
- [ ] Temporary files are removed after successful, failed, and overloaded requests.
- [ ] Run a 100-request staging soak test with no memory growth, worker restart, or increasing latency trend.
- [ ] Save the new load-test report and compare it with the previous 43-second back and 64-second front medians.

## 8. Production rollout

- [ ] Confirm a recent server snapshot or machine-image rollback point exists.
- [ ] Keep the previous application build and configuration available.
- [ ] Drain traffic from the old instance gracefully.
- [ ] Deploy the new commit to staging and complete Sections 5–7.
- [ ] Deploy to production during a low-traffic period.
- [ ] Send 5–10% canary traffic first when the infrastructure supports it.
- [ ] Verify one front and one back request through the real client-facing URL.
- [ ] Watch application errors, response time, CPU, RAM, GPU utilization, GPU memory, disk, and HTTP 503 rate for at least 30 minutes.
- [ ] Increase traffic only when metrics and extracted-field accuracy remain acceptable.
- [ ] Record the final production commit and configuration values in the release record.

## 9. Rollback triggers

Rollback immediately if any of the following occurs:

- [ ] CUDA is unavailable or logs show unintended CPU inference.
- [ ] CUDA out-of-memory occurs, the worker crashes, or the service repeatedly restarts.
- [ ] Critical-field accuracy is worse than the previous release.
- [ ] HTTP 5xx error rate exceeds 2% for 10 minutes under normal, non-overload traffic.
- [ ] Two-user P95 latency exceeds 90 seconds for 10 minutes.
- [ ] GPU memory remains above 15 GB, RAM remains above 90%, or swap activity continues under normal load.
- [ ] Temporary NID files remain after requests or personal data appears in logs.
- [ ] The client cannot complete either the front or back critical workflow.

## 10. Rollback procedure

- [ ] Stop sending new traffic to the affected instance.
- [ ] Capture application and GPU logs without copying client NID images.
- [ ] Restore the previous stable commit and its matching environment configuration.
- [ ] Restart with one worker and wait for the previous model warm-up/readiness check.
- [ ] Verify one known front image and one known back image.
- [ ] Restore traffic and monitor for at least 30 minutes.
- [ ] Document the failure, affected period, rollback time, and follow-up owner.

## Deployment approval

- [ ] Technical checks approved by: `____________________`
- [ ] Accuracy checks approved by: `____________________`
- [ ] Production owner approval: `____________________`
- [ ] Deployment result: `approved / rolled back / postponed`
- [ ] Notes: `____________________________________________________________`
