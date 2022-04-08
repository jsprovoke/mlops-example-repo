#!/bin/bash

git clone https://jsprovoke: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@github.com/jsprovoke/dvc-example-data

cd dvc-example-data
dvc remote modify origin --local auth basic
dvc remote modify origin --local user jsprovoke
dvc remote modify origin --local password 2cf70b9982f050dcf6dd9dc04731a1d39fa48988

git remote add origin https://jsprovoke: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@github.com/jsprovoke/dvc-example-data.git

dvc pull -r origin

mkdir output

cd ..

python3 run.py

cd dvc-example-data

dvc add output

dvc push -r origin

git add .

git config --global user.email "johns@provoke.co.nz"

git commit -m "Output files added to DVC tracked repo"

git push -u origin
