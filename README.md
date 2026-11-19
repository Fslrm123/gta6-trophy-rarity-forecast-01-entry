# GTA VI Trophy Rarity Forecast

Entry for the make.fun bounty [Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): a Python 3.13 program (standard library only) that reads the house's recording of Exophase's GTA VI PS5 trophy page and forecasts every trophy's earn rate at the case target as five quantiles.

The model treats an earn rate as a ratio — earners over players tracked — and fits a separate log-log growth curve to each count over the distinct page states in the history, so Exophase's refresh jumps (players up, rates diluted) are handled by the decomposition rather than fought as noise. The earners trend is damped toward the players trend past the last read, since rates saturate. Spreads are logit-normal, built from a random-walk floor, the model's own residuals, and an extrapolation term.

![bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)
![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Fslrm123/gta6-trophy-rarity-forecast-01-entry/rank.svg)
![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Fslrm123/gta6-trophy-rarity-forecast-01-entry/reward.svg)

Run locally with `mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py`.

New to make.fun? Join with my referral link: https://make.fun/r/3v2j7j85
