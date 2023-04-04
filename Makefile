.PHONY: start
start:
	uvicorn main:app --reload --port 9000 --env-file .env --proxy-headers --forwarded-allow-ips="*"

.PHONY: format
format:
	black .
	isort .