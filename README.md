# Proyecto Redes Neuronales - face2face

El proyecto es un fork de [face2face-demo](https://github.com/datitran/face2face-demo) de [datitran](https://github.com/datitran). Utiliza pix2pix para aprender los rasgos faciales y ponerlos en una cara. Por medio de una webcam o un video, coloca tu rostro en la cara "entrenada" de EPN en tiempo real o no.

Debido al pobre poder de computo de mi laptop, correr el programa en tiempo real fue imposible, así que agregué [run_video.py](https://github.com/cesern/face2face-demo/blob/master/run_video.py) tomado de [face2face-demo](https://github.com/karolmajek/face2face-demo) de [karolmajek](https://github.com/karolmajek) .

## Getting Started

#### 1. Preparando environment

```
# Clonar est repositorio
git clone git@github.com:datitran/face2face-demo.git

# Crear el environment desde el archivo
conda env create -f environment.yml

# Activar environment
source activat face2face-demo
```

#### 2. Generar datos de entrenamiento

```
python generate_train_data.py --file peñaRecorte.mp4 --num 400 --landmark-model shape_predictor_68_face_landmarks.dat
```

Input:

- `file` es el nombre del video con el que se creara el data set.
- `num` es el número de datos de entrenaminto que seran creados.
- `landmark-model` es el  facial landmark model usado para detectar landmarks. Un pre-entrenado facial landmark model esta incluido en el repositorio.

Output:

- Dos folders `original` and `landmarks` seran creados.

#### 3. Entrenar modelo

```
# Clonar este repositorio de Christopher Hesse's pix2pix TensorFlow implementation
git clone https://github.com/affinelayer/pix2pix-tensorflow
cd pix2pix-tensorflow

# Reset a la versión de abril
git reset --hard d6f8e4ce00a1fd7a96a72ed17366bfcb207882c7

# Mover al directorio de pix2pix y crear una carpeta para las fotos
mkdir photos

# Mover las carpetas original y landmarks a la carpeta de pix2pix-tensorflow
mv face2face-demo/landmarks face2face-demo/original pix2pix-tensorflow/photos

# Ve al folder de pix2pix-tensorflow
cd pix2pix-tensorflow/

# Redimensionar las imagenes de original
python tools/process.py \
  --input_dir photos/original \
  --operation resize \
  --output_dir photos/original_resized
  
# Redimensionar las imagenes de landmark
python tools/process.py \
  --input_dir photos/landmarks \
  --operation resize \
  --output_dir photos/landmarks_resized
  
# Combinar las imagenes de original y landmark
python tools/process.py \
  --input_dir photos/landmarks_resized \
  --b_dir photos/original_resized \
  --operation combine \
  --output_dir photos/combined
  
# Dividir en train/val
python tools/split.py \
  --dir photos/combined
  
# Entrenar el modelo
python pix2pix.py \
  --mode train \
  --checkpoint name \
  --output_dir face2face-model \
  --max_epochs 200 \
  --input_dir photos/combined/train \
  --which_direction AtoB
  --save_frequ 50
```

Christopher Hesse's [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow) implementation.

#### 4. Exportar el modelo

1. Se necesita reducir el modelo entrenado.
    ```
    python reduce_model.py --model-input face2face-model --model-output face2face-reduced-model
    ```
    
    Input:
    
    - `model-input` la carpeta del modelo.
    - `model-output` la carpeta donde quedará el modelo reducido.
    
    Output:
    
    - Regresa un modelo reducido con un tamaño menor que el original

2. Se congela el modelo reducido a un solo archivo.
    ```
    python freeze_model.py --model-folder face2face-reduced-model
    ```

    Input:
    
    - `model-folder` la carpeta donde esta el modelo reducido.
    
    Output:
    
    - `frozen_model.pb` en la carpeta del modelo.
    
Pre-trained frozen model [Proximamente](). Modelo entrenado con 400 imagenes y 200 epoch.
    
#### 5. Run Demo

Real-Time
```
python run_webcam.py --source 0 --show 0 --landmark-model shape_predictor_68_face_landmarks.dat --tf-model face2face-reduced-model/frozen_model.pb
```

Input:

- `source` is the device index of the camera (default=0).
- `show` mostrar la imagen de la webcam (0) o los rasgos faciales (1) (default=0).
- `landmark-model`  facial landmark model usado apra detectar rasgos faciales.
- `tf-model` el modelo congelado.

NO Real-Time
```
python run_video.py --source video.mp4 --show 0 --landmark-model shape_predictor_68_face_landmarks.dat --tf-model face2face-reduced-model/frozen_model.pb
```

Input:

- `source` video a usar (default=0).
- `show` mostrar la imagen de la webcam (0) o los rasgos faciales (1) (default=0).
- `landmark-model`  facial landmark model usado apra detectar rasgos faciales.
- `tf-model` el modelo congelado.

Ejemplo original:

![example](example.gif)

Resultado:

![ejemplo](ejemplo.gif)

## Requerimientos
- [Anaconda / Python 3.5](https://www.continuum.io/downloads)
- [TensorFlow 1.2](https://www.tensorflow.org/)
- [OpenCV 3.0](http://opencv.org/)
- [Dlib 19.4](http://dlib.net/)
