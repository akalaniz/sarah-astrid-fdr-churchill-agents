# Durable Engineering Rules

- Sarah and Astrid are existing persistent agents. Reuse their real `app.core.sarah_engine` paths; do not replace them with prompt-only substitutes.
- Space Invaders lives under `games/space_invaders`; its authoritative human game is `sarah_astrid_ooda_game_v3_focusfix.py`.
- Preserve continuous game timing. Movement, firing, and LLM latency must never gate simulation time.
- LLM/API calls must never run on or block the Tkinter thread, and only one game decision call may be outstanding.
- Human keyboard mode must remain operational, including firing while moving.
- Preserve the SPACE/Restart fix: after restart, SPACE fires and never activates Restart.
- Run the focused Space Invaders tests and relevant Sarah/Astrid web/chat regressions after game changes.
