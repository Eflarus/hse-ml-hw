import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from io import StringIO
from pprint import pprint
import re

# init FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
MODEL_PATH = "./models/model_hw1_full.pkl"
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    raise Exception(f"model not found at {MODEL_PATH}")


# preprocess
def mileage_to_kmpl(value):
    if pd.isna(value):
        return np.nan
    try:
        mileage, unit = value.split()
        mileage = float(mileage)
        if unit == 'kmpl':
            return mileage
        elif unit == 'km/kg':
            # 1 kg ~ 1.5 литра топлива
            return mileage * 1.5
        else:
            return np.nan
    except Exception:
        return np.nan


def engine_to_cc(value):
    if pd.isna(value):
        return np.nan
    try:
        engine_value = value.split()[0]
        return float(engine_value)
    except Exception:
        return np.nan


def max_power_to_bhp(value):
    if pd.isna(value):
        return np.nan
    try:
        power_value = value.split()[0]
        return float(power_value)
    except Exception:
        return np.nan


def process_torque(value):
    if pd.isna(value):
        return np.nan, np.nan
    try:
        value = value.replace(" ", "").replace(",", "").lower()
        torque_match = re.search(r"([\d.]+)", value)
        if not torque_match:
            return np.nan, np.nan
        torque_value = float(torque_match.group(1))
        if "kgm" in value:
            torque_value *= 9.8  # kgm to Nm
        rpm_match = re.search(r"(?:@|at)([\d,-]+)", value)
        if rpm_match:
            rpm_value = rpm_match.group(1)
            if "-" in rpm_value:
                max_rpm = float(rpm_value.split("-")[-1])
            else:
                max_rpm = float(rpm_value)
        else:
            max_rpm = np.nan
        return torque_value, max_rpm
    except Exception as e:
        pprint(e)
        return np.nan, np.nan


def process_name(df):
    df['name'] = df['name'].str.lower()
    print(f"{df['name'].nunique()=}")
    df[['brand', 'model', 'submodel']] = df['name'].str.split(' ', n=2, expand=True)
    df['drive_type'] = (df['submodel']
                        .str.extract(r'(4x4|4wd|2wd|4x2|2wd)', expand=False)
                        .fillna('2wd')
                        .replace('4wd', '4x4')
                        )
    df['engine_w'] = (df['submodel']
                      .str.extract(r'(w\d+)', expand=False)
                      .fillna('no')
                      )
    df['comp'] = (df['submodel']
                  .str.extract(r'(\b[a-z]{2}i\b)', expand=False)
                  .fillna('no')
                  )
    df['submodel'] = (df['submodel']
                      .str.replace(' at', '')
                      .str.replace('at ', '')
                      .str.replace(r'[()/\-+]', '', regex=True)
                      .str.replace(r'(4x4|4wd|2wd|4x2|2wd|mt|r|vvt)', '', regex=True)
                      .str.replace(r'\b[a-zA-Z]{2}i\b', '', regex=True)
                      .str.replace(r'\bw\d+\b', '', regex=True)
                      .str.replace(r'\b\d+(\.\d+)?\b', '', regex=True)
                      .str.replace(')', '')
                      .str.replace('  ', ' ')
                      .str.strip()
                      .apply(lambda x: ' '.join(sorted([word for word in x.split()])))
                      )
    return df


def process_df_to_normal_coltypes(df):
    df['mileage'] = df['mileage'].apply(mileage_to_kmpl).astype(float)
    df['engine'] = df['engine'].apply(engine_to_cc).astype(int)
    df["seats"] = df["seats"].astype(int)
    df['max_power'] = df['max_power'].apply(max_power_to_bhp).astype(float)
    df[['torque', 'max_torque_rpm']] = df['torque'].apply(
        lambda x: pd.Series(process_torque(x))
    )
    df = process_name(df)
    return df


# Pydantic-классы для описания данных
class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


categorical_features = [
    'brand', 'model',
    'drive_type',
    'engine_w',
    # 'comp',
    'year', 'fuel', 'seller_type',
    'transmission', 'owner', 'seats']

numerical_features = [
    'mileage',
    'engine',
    'max_power', 'torque',
    # 'max_torque_rpm',
    'km_driven'
]


# POST-метод для предсказания одного объекта
@app.post("/predict_item")
def predict_item(item: Item) -> float:
    # Преобразование входных данных в DataFrame
    df = pd.DataFrame([item.model_dump()])
    df = process_df_to_normal_coltypes(df)

    # nat_cols = ['mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm', 'seats']
    # X = df[nat_cols]
    X = df[categorical_features + numerical_features]

    try:
        prediction = model.predict(X)
        return float(prediction[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)):
    try:
        content = file.file.read().decode("utf-8")
        df = pd.read_csv(StringIO(content))

        items = []
        invalid_rows = []
        for index, row in df.iterrows():
            try:
                item = Item(
                    name=row["name"],
                    year=row["year"],
                    selling_price=row["selling_price"],
                    km_driven=row["km_driven"],
                    fuel=row["fuel"],
                    seller_type=row["seller_type"],
                    transmission=row["transmission"],
                    owner=row["owner"],
                    mileage=row["mileage"],
                    engine=row["engine"],
                    max_power=row["max_power"],
                    torque=row["torque"],
                    seats=row["seats"],
                )
                items.append(item)
            except Exception as e:
                invalid_rows.append((index, str(e)))

        if invalid_rows:
            raise HTTPException(
                status_code=400,
                detail=f"Validation failed for rows: {invalid_rows}",
            )

        df = pd.DataFrame([item.model_dump() for item in items])

        df = process_df_to_normal_coltypes(df)

        # nat_cols = ['mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm', 'seats']
        # X = df[nat_cols]
        X = df[categorical_features + numerical_features]

        df['predicted_price'] = model.predict(X)

        output = StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return Response(output.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=predictions.csv"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
