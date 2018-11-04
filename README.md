# Proyecto Redes Neuronales - face2face

El proyecto es un fork de [face2face-demo](https://github.com/datitran/face2face-demo). Utiliza pix2pix...
Es una aplicación de webcam, que coloca tu rostro en la cara "entrenada" de EPN en tiempo real.

## Getting Started

#### 1. Preparando environment

```
# Clonar est repositorio
git clone git@github.com:datitran/face2face-demo.git

# Create el conda environment desde el archivo
conda env create -f environment.yml

# Activar environment
source activat face2face-demo

# Instalar en el environment
pip install opencv-python
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

# Reset to april version
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
  --output_dir face2face-model \
  --max_epochs 200 \
  --input_dir photos/combined/train \
  --which_direction AtoB

...