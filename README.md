# BBTX11-VT26-07: Classifying Cellular States from Gene Expression with Interpretable Machine Learning

## Instruktioner för användande av Chalmers Minerva cluster
> Note: Full guide finns på: **https://git.chalmers.se/karppa/minerva**
**OBS** Bara tillgängligt från Chalmers nätverk (om man inte använder VPN)

## Steg
1. Ansök om access till Chalmers Minerva cluster ![here](https://forms.office.com/e/NLe5HDPGKY). Det tar vanligtvis några dagar att få svar. 

2. (**obs bara första gången**) Input ```ssh CID@minerva.cse.chalmers.se ``` och ditt CID-lösenord i terminalen för att aktivera tillgång. 

3. Installera VSCode, med ![Remote - SSH Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) extension.

4. Logga in i Chalmers kluster via Remote SSH i VSCode, med CID@minerva.cse.chalmers.se och ditt eget CID-lösenord (obs. välj Linux oavsett eget operativsystem, då klustret använder det)

5. Inne i klustret (VSCode öppnar nytt fönster, ljusblå bar i nedre vänstra hörnet) skapar du en fil som heter något i stil med **my-script.sh** som innehåller: 

```
#! /bin/bash

#SBATCH --job-name=KulingNet-testjob
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu

source /data/users/thomath/"Kandidatarbete BBTX11"/.venv/bin/activate

python3 KulingNet3.0sshtest.py 
```

6. 