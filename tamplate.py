import os
import logging
from pathlib import Path 

project_name = "crop_yield_prediction"

log_dir = "logs"
log_file = "project_setup.log"
os.makedirs(log_dir , exist_ok = True)
logging.basicConfig(
    
    filename= os.path.join(log_dir , log_file),
    level= logging.INFO , 
    format= "[%(asctime)s] %(levelname)s - %(massage)s", 
    encoding= "utf-8"
    
)

logger = logging.getLogger(__name__)

files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/configuration/__init__.py",
    f"src/{project_name}/configuration/config.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_preprocessing.py",
    f"src/{project_name}/components/data_training.py",
    f"src/{project_name}/components/data_evalution.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/stage_01_data_ingestion.py",
    f"src/{project_name}/pipeline/stage_02_data_validation.py",
    f"src/{project_name}/pipeline/stage_03_data_preprocessing.py",
    f"src/{project_name}/pipeline/stage_04_data_training.py",
    f"src/{project_name}/pipeline/stage_05_data_evalution.py",
    "config/config.yaml",
    "params.yaml",
    "main.py",
    "requirements.txt",
    ".gitignore",
    "setup.py",
    "app.py"
]

if __name__ =="__main__":
    for file_path in files:
        file_path = Path(file_path)
        file_dir = file_path.parent
        
        if file_dir and not file_dir.exists():
            file_dir.mkdir(parents=True, exist_ok= True)
            logger.info(f"{file_dir} directory created successfully !!")
            
        if not file_path.exists():
            file_path.touch()
            logger.info(f"{file_path} file created successfully !!")