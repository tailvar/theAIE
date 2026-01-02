# Runbook: staging-api CrashLoopBackOff

## Quick triage
1. List pods:
   - kubectl get pods -n staging
2. Inspect previous container logs:
   - kubectl logs <pod> -n staging --previous
3. Describe pod for events:
   - kubectl describe pod <pod> -n staging

## Common causes
- bad env var / secret missing
- incompatible image tag
- migration running at boot and failing
- memory limit too low / OOMKill

## Mitigations
- rollback last deploy
- increase memory limit temporarily
- disable failing feature flag if applicable
