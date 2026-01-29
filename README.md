# BBTX11-VT26-07: Classifying Cellular States from Gene Expression with Interpretable Machine Learning

## Instruktioner för förstagångs-användande av Chalmers Minerva cluster
> Note: Full guide finns på: **https://git.chalmers.se/karppa/minerva**

**OBS** Bara tillgängligt från Chalmers nätverk (om man inte använder VPN)

## Innan man kan använda det
1. Ansök om access till Chalmers Minerva cluster [here](https://forms.office.com/e/NLe5HDPGKY). Det tar vanligtvis några dagar att få svar. 

2. Input ```ssh CID@minerva.cse.chalmers.se ``` och ditt CID-lösenord i terminalen för att aktivera tillgång. 

3. Gå till ditt valda directory och skapa en mapp för virtual environment med ```python3 -m venv .venv```. 

4. Gå in i ditt venv. med ```source .venv/bin/activate```, nu borde ```(.venv)``` stå till vänster i din terminal. 
 
5. Installera alla packages som projektet behöver med pip, ex. ```pip install numpy```, dessa hamnar nu i venv och kan återanvändas lätt vid framtida körningar. 

6. Installera VSCode, med [Remote - SSH Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) extension.

7. Logga in i Chalmers kluster via Remote SSH i VSCode, med CID@minerva.cse.chalmers.se och ditt eget CID-lösenord (obs. välj Linux oavsett eget operativsystem, då klustret använder det)

8. Inne i klustret (VSCode öppnar nytt fönster, ljusblå bar i nedre vänstra hörnet) skapar du en fil som heter något i stil med **my-script.sh** som innehåller: 

```
#! /bin/bash

#SBATCH --job-name=namn-på-job
#SBATCH --cpus-per-task=2

source /path/to/your/.venv/bin/activate

python3 namn-på-programfil.py 
```

med placeholders utbytta mot verkliga namn. **OBS** Använd #SBATCH --gres=gpu för att köra på GPU om det verkligen behövs. 

9. Du bör nu kunna köra din batch-fil med ```sbatch my-script.sh``` och få all output till en fil i stil med ```slurm-117668.out```


