# Navod na setup
Ako prvé treba nainštalovať Conda prostredie 
Conda je nástroj, ktorý nám dovolí vytvárať izolované virtuálne prostradie pre python. Disponuje vlastným správcom balíkov a taktiež si vieme pre každé prostredie flexibilne meniť python verziu.
Download link: [conda](https://www.anaconda.com/download/success)

Prepneme sa do adresáru projektu
```shell
cd path/to/project
```

vytvorenie prostredia spolu s potrebnými kniznicami
```sh
conda env create -f environment.yml
```

aktivácia vytvoreného prostedia 
```shell
conda activate opencv-env
```

-main.py je hlavny skript
