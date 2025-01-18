


#you need to put code specific to getting colab up and runnign here, like where any stuff on google drive is and loading google drive etc...
'''

def mount_drive(force=False):
    #mount google drive
    drive_base_path = '/content/drive'
    drive.mount(drive_base_path, force_remount=force)

    
if not local_flag:
    drive_path = '/content/drive/My Drive/Colab Notebooks/Spert-nat/conll04 - spert'
    colab_path = '/content/dataset'
    main_path = '/content'

    !pip install -q pyarrow==14.0.2 transformers[torch] datasets evaluate

    from google.colab import drive    #only import this when on colab
    # Mount Google Drive
    drive.mount('/content/drive')
    os.makedirs(colab_path, exist_ok=True)
    for k,v in files.items():
        srce_path = drive_path + '/' + v
        dest_path = colab_path + '/' + v
        !cp -f '{srce_path}' '{dest_path}'


    
    #use this if you want to run the real spert, just download the diles to the colab machine and run it....
    #temp
    #!cp -fr '/content/drive/My Drive/Colab Notebooks/Spert-nat/spert-master' '/content/spert-master'
    #then run this in a terminal:
    #cd spert-master
    #python ./spert.py train --config configs/example_train.conf
    
#do this for local run
else:
    import os, subprocess
    main_path = 'D:/A.Nathan/1a.UWA24-Hons/Honours Project/0a.Code/Spert-nat'
    os.chdir(main_path)
    colab_path = 'D:/A.Nathan/1a.UWA24-Hons/Honours Project/0a.Code/Spert-nat/conll04 - spert'
'''