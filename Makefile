.PHONY: start
start:
	uvicorn main:app --reload --host 0.0.0.0 --port 9000 --env-file .env --proxy-headers --forwarded-allow-ips="*"

.PHONY: format
format:
	black .
	isort .