PROJECT := $(CURDIR)
PYTHON := $(PROJECT)/.venv/bin/python
export PYTHONPATH := $(PROJECT)/backend
export PYTHONDONTWRITEBYTECODE := 1

.PHONY: api frontend infra migrate worker scheduler test check sync sync-index sync-gpu
.PHONY: start stop status diagnose demo backup restore
start:
	"$(PYTHON)" backend/scripts/manage.py start --scheduler
stop:
	"$(PYTHON)" backend/scripts/manage.py stop
status:
	"$(PYTHON)" backend/scripts/manage.py status
diagnose:
	"$(PYTHON)" backend/scripts/manage.py diagnose
demo:
	"$(PYTHON)" backend/scripts/manage.py start --profile demo --readonly --api-port 18083 --frontend-port 4174
backup:
	"$(PYTHON)" backend/scripts/backup_restore.py backup $(ARGS)
restore:
	"$(PYTHON)" backend/scripts/backup_restore.py restore $(ARGS)
api:
	"$(PYTHON)" -m uvicorn research_loop.main:app --host 127.0.0.1 --port 18082
frontend:
	cd frontend && npm run dev
infra:
	docker compose up -d
migrate:
	"$(PYTHON)" backend/scripts/migrate.py
worker:
	"$(PYTHON)" backend/scripts/run_worker.py
scheduler:
	"$(PYTHON)" backend/scripts/run_scheduler.py
test:
	"$(PYTHON)" -m unittest discover -s backend/tests -v
	cd frontend && npm test
check:
	"$(PYTHON)" -c 'import ast,pathlib; [ast.parse(p.read_text(), filename=str(p)) for p in pathlib.Path("backend").rglob("*.py")]'
	cd frontend && npm run check
sync:
	UV_PROJECT_ENVIRONMENT="$(PROJECT)/.venv" uv sync --project backend --locked
sync-index:
	UV_PROJECT_ENVIRONMENT="$(PROJECT)/.venv" uv sync --project backend --locked --extra local-index

sync-gpu:
	UV_PROJECT_ENVIRONMENT="$(PROJECT)/.venv" uv sync --project backend --locked --extra local-index --extra apple-gpu
