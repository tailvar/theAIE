# CrashLoopBackOff Runbook

If pods are in CrashLoopBackOff:
1) Check recent deployment changes (image tag, env vars, secrets).
2) Inspect logs: 'kubetcl logs <pod> --previous'
3) Describe pod: 'kubectl describe pod <pod>'
4) Common causes: missing secrets, bad config, failing healthcheck, image pull issues.
5) If safe, roll back deployment or restart pods after identifying root cause.