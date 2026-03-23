-----------------------------------------
Lucas ReadMe, uppdaterad söndag 8/3 19:00
-----------------------------------------

Detta är det jag pillat med hittills och nu leker jag allaballan och gör min egen readme lol.

* "ReactomePathways.gmt", "ReactomePathways.txt" och "ReactomePathwaysRelationship.txt" är datan jag laddat ned,
    från https://reactome.org/download-data.

* "csvconverterattempt2.py" läser in filerna och slår ihop alla pathways + relationer till en .csv-fil, 
    där noderna visar sin child-parent relation. 
    
* "binn_connectivity.csv": Resultatet av konverteringen. Denna innehåller hela hierarkin.
* "hda5tocsv.py": Ett litet hjälpskript för att kunna kika på innehållet i .h5ad-filer utan att öppna dem i Python.

* "MaskCreatorV1.py" tar indatan (från conv_data) och gör masking layers för varje celltyp var för sig.
    (genom att mappa generna mot relations-datan i "binn_connectivity.csv")
Resultatet sparas i mappen "MaskMatrixLayers". Varje celltyp har 4 lager av masker (layer_0 till layer_3). Ändra antal lager sen?

* Lager 0: Kopplar våra 33k+ gener till de minsta pathwaysen.
* Lager 1-3: Kopplar ihop pathways till större och större biologiska processer.
 
Nollor i matrisen betyder "ingen biologisk koppling", ettor betyder att en koppling finns. Implementera dessa mha pytorch nu?
Kommer såklart byta namn på mappen/filerna sen, döpte bara t PathwaysLA just nu för att förtydliga att detta inte är komplett.
Simons grejer ligger i simon_test.
