# Sort GIT out
Explaining GIT history @ CheffTreffHackathon 

> Developed in <24 hours on 24-25. Apr 2025

## Features
### `main` (solution by me, @eburakova)
   - Analyses commit messages and difference logs for each file changed in the commit.
   - Stores the messages and difference files along with their generated summarries as a local json database.
   - The length of history depends on the amount of tokens the LLM is able to process at once.
   - Simple, sleek UI (made with [Streamlit](https://streamlit.io/))
   - AI base model: Only Gemini is supported.
### Other branches
Do check out solutions by my talented teammates! 
  - Insteadd of the local database, rely on GitHub API
  - Spend more Gemini tokens but work faster than `main`
  - Just as simple and sleek [Streamlit](https://streamlit.io/) UI

## Future development
**Do feel free to fork and build up!**
### TODO: 
- Add support for more model providers, i.e. OpenAI
- Add support for local LLMs
>[!NOTE]
>Connecting to a local LLMs may be desirable for privacy of the codebase, but smaller models *will* yield worse results!

## Tested on 
- [Linux kernel repo](https://github.com/torvalds/linux) (20000 latest commits)
- [ohmyzsh](https://github.com/ohmyzsh/ohmyzsh)
- Test repos provided by FinanzInformatik team: [Calculator](https://github.com/Bl7tzcrank/ChefTreffHackFIChallenge#) and [Git guidelines](https://github.com/RomuloOliveira/commit-messages-guide/)
