# Running the pipeline script
## **Change diretory to project root**
```
cd nlp_cia
```
## **install using poetry**
```
poetry install
```
## **Download model from:**
Onedrive, [Models_nlp_2526](https://edubuas-my.sharepoint.com/:f:/g/personal/225538_buas_nl/EuvOvnKbcLdPnsq8LGqDbJ0BP0DLEER7PMAwyS1bOy4JZg?e=olbDPM)
## **Place model in:**
`nlp_cia/models`
## **Run pipeline using**
```
poetry run python src/pipeline.py
```
## **When prompted paste link into command line**
# Running notebooks
## **Change directory to project root**
```
cd nlp_cia
```
## **Create jupyter kernel from poetry enviroment**
```
poetry run python -m ipykernel install --user --name <your-kernel-name>
```
## **Restart code editor**