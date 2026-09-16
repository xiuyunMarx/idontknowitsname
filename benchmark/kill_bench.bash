#!/bin/bash
# Stop every benchmark process: scan chain, sweeps, drivers, agent sessions, server.
pkill -f "[f]ind_regime.bash"; pkill -f "[s]weep_(BFCL|coding|finance|fact_check)"; pkill -f "[m]ixed_workload"
pkill -f "[j]ac run"; sleep 2; pkill -9 -f "[s]glang::|[s]erver\.(server|continuum_server|cacheScout_server)"; sleep 4
pgrep -af "regime|sweep_|mixed_work|jac run|server\.(server|continuum_server|cacheScout_server)|sglang::" | grep -v pgrep | wc -l
