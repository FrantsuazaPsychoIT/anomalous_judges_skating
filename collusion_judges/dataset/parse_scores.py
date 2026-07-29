# collusion_judges/dataset/parse_scores.py
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests
import typer
from bs4 import BeautifulSoup, Tag
from loguru import logger
from tqdm import tqdm

from collusion_judges.config import (
    DEFAULT_HEADERS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REQUEST_SLEEP,
    REQUEST_TIMEOUT,
)

# -------------------- Constants (domain-specific) --------------------


BASE_URL = "https://skatingscores.com"

EVENTS_2022 = ["2223/gpusa", "2223/gpcan", "2223/gpfra", '2223/gpgbr', "2223/gpjpn",  "2223/gpfin", '2223/gpf']
EVENTS_2023 = ["2324/gpusa", "2324/gpcan", "2324/gpfra", '2324/gpchn', "2324/gpfin", "2324/gpjpn", "2324/gpf"]
EVENTS_2024 = ["2425/gpusa", "2425/gpcan", "2425/gpfra", "2425/gpjpn", "2425/gpfin", '2425/gpchn', "2425/gpf"]


EVENTS: list[str] = EVENTS_2022 + EVENTS_2023 + EVENTS_2024
GENDERS: list[str] = ["men", "ladies", "pairs", "dance"]
PROGRAMS: list[str] = ["short", "long"]

N_JUDGES: int = 9

EVENT_TO_COUNTRY: dict[str, str] = {
    "2022_GP_Espoo": "FIN",
    "2022_GP_MK_John_Wilson_Trophy": "GBR",
    "2022_GP_NHK_Trophy": "JPN",
    "2022_GP_Skate_America": "USA",
    "2022_GP_Skate_Canada": "CAN",
    "2022_GP_de_France": "FRA",
    "2022_Grand_Prix_Final": "ITA",
    "2023_GP_Cup_of_China": "CHN",
    "2023_GP_Espoo": "FIN",
    "2023_GP_NHK_Trophy": "JPN",
    "2023_GP_Skate_America": "USA",
    "2023_GP_Skate_Canada": "CAN",
    "2023_GP_de_France": "FRA",
    "2023_Grand_Prix_Final": "CHN",
    "2024_GP_Cup_of_China": "CHN",
    "2024_GP_Finlandia_Trophy": "FIN",
    "2024_GP_NHK_Trophy": "JPN",
    "2024_GP_Skate_America": "USA",
    "2024_GP_Skate_Canada_Intl": "CAN",
    "2024_GP_de_France": "FRA",
    "2024_Grand_Prix_Final": "FRA",
}

app = typer.Typer(add_completion=False)

BASE_URL = "https://skatingscores.com"
REQUEST_SLEEP = 0.2
REQUEST_TIMEOUT = 30


# ----------- 1. Скачиваем данные со страниц -----------------------------
# -------------------- Fetching --------------------


def build_urls(
    base_url: str,
    events: Iterable[str],
    genders: Iterable[str],
    programs: Iterable[str],
) -> list[str]:
    """Build page URLs for (event, gender, program)."""
    return [
        f"{base_url}/{event}/{gender}/{program}"
        for event in events
        for gender in genders
        for program in programs
    ]

def fetch_html(
    url: str,
    *,
    sleep_seconds: float = REQUEST_SLEEP,
    timeout_seconds: float = REQUEST_TIMEOUT,
    headers: Optional[dict[str, str]] = None,
) -> bytes:
    """Download HTML bytes for a URL."""
    time.sleep(sleep_seconds)
    r = requests.get(url, timeout=timeout_seconds, headers=headers or DEFAULT_HEADERS)
    r.raise_for_status()
    return r.content

def fetch_soup(url: str) -> BeautifulSoup:
    """Download and parse page into BeautifulSoup."""
    html = fetch_html(url)
    return BeautifulSoup(html, "html.parser")



def fetch_html(url):
    # https://skatingscores.com/2223/gpchn/jr/men/short/

    time.sleep(0.2)  # to avoid overloading the skatingscores server
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.content
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)


def fetch_soups(urls):
    return [BeautifulSoup(fetch_html(url), 'html.parser') for url in urls]

def fetch_soup(url):
    return BeautifulSoup(fetch_html(url), 'html.parser')





# ----------- 2. Переводим данные в более удобный формат -----------------

#  На вход: страница мероприятия
#  На выход: (год, этап, м/ж, кп/пп)
def get_event_name(soup):
    event = re.sub(r'[^A-Za-z0-9 ]+', '',  soup.find("h1").text).lstrip().split()
    if event[-5] == 'GP':
        return event[0], ' '.join(event[1: -5]),  event[-3], event[-2] + event[-1]
    return event[0], ' '.join(event[1: -4]),  event[-3], event[-2] + event[-1]



#  На вход: страница мероприятия
#  На выход: skate_soups: список из данных по каждому спортсмену

def get_skates(soup):
    return soup.find_all("div", {"class": "skat-wrap"})

#  На вход: skate_soup: данные по 1 спортсмену
#  На выход: list из 2 таблиц: техника + компоненты


# ----------- 3. Собираем общие данные по фигуристам и судьям ------------


def get_all_points(skate_soup):
    return skate_soup.find_all("div", {"class": "ptab2-wrap"})

# def get_technique(skate_soup):
#    return skate_soup.find("div", {"class": "ptab2-wrap"})

# def get_components(skate_soup):
#    return skate_soup.find("div", {"class": "ptab2-wrap"})


def get_juries_names_and_country(skate_soup):
    row = skate_soup.find("div", {"class": "ptab1-wrap"}).find("table", {'class' : 'ptab2'}).find("tr", {"class": "tally"})
    row = row.find_all('td', {'class' : 'c'})
    names = [td.find('a', class_='jlink').attrs['title'] for td in row]
    countries = [td.text.split()[0] for td in row]
    return names, countries

def get_name_and_country(skate_soup, url):
    headlinks = (
        skate_soup
        .find("div", class_="ptab1-wrap")
        .find("table")
        .find("tr", class_="head")
    )
    country = headlinks.find_all('td')[2].text.split()[0]

    if 'pairs' in url or 'dance' in url:
        #  name = " ".join([n.capitalize() for n in headlinks.find_all("a")[0].text.split(" ")])
        spans = headlinks.find_all("a")[0].find_all("span")
        names = [' '.join(list(map(lambda x: x.capitalize(), s.get_text(strip=True).split()))) for s in spans if s.get_text(strip=True)]

        # имена пары будут списком, объединим их через запятую или через " / "
        name = " / ".join(names)
    else:
        name = " ".join([n.capitalize() for n in headlinks.find_all("a")[0].text.split(" ")])

    return name, country

# ----------- 4. Получаем оценки каждого фигуриста -----------------------

def get_base_value(skate_soup) -> float:
    base_value = skate_soup.find('td', class_ = 'r eltot').text
    match = re.search(r'(\d+\.\d+)', base_value)
    return float(match.group(1))

#  Вот из такой фигни я добываю 42.95
#  <td class="r eltot"><span class="sbt">SB</span>&nbsp; 2 <br> 42.95</td>

def get_base_values(table_score_soup):
    base_values = [td.text for td in table_score_soup.find("table").find("tr", {"class": "elrow"}).find_all("td", {"class" : "r __100__"})]
    return base_values

def get_place_and_score(score_list):
    score_list = list(map(lambda x: x.split(), score_list))
    places = [int(place_score[0].strip()) for place_score in score_list]
    scores = [float(place_score[1].strip()) for place_score in score_list]
    return places, scores


#  Функция возвращает финальные оценки судей одного спортсмена
#  Вход:
#  Выход: score_list

def get_scores_places(table_score_soup):
    table = [td.text for td in table_score_soup.find("table").find("tr", {"class": "tally"}).find_all("td", {"class" : "c"})]

    #table = [td.text for td in  all_points[1].find('tr', {'class' : 'tally'}).find_all('td')[1:]]
    return get_place_and_score(table)



#  Функция возвращает финальные оценки судей одного спортсмена
#  Вход:
#  Выход: score_list

def get_elems_evaluation(table_score_soup):
    table = [[td.text for td in tr.find_all('td', {'class' : 'c'})] for tr in table_score_soup.find("table").find("tr", {"class": "elrow"})]

    return table

# ----------- 5. Создание датасета ---------------------------------------
def get_dataset(data, urls):

    for url in urls:  # Итерация по страницам
        print(url)
        soup = fetch_soup(url)
        skates_soup = get_skates(soup)
        year, event, gender, program = get_event_name(soup)

        for skate_soup in skates_soup:
            base_value = get_base_value(skate_soup)
            name, country = get_name_and_country(skate_soup, url)
            names, countries = get_juries_names_and_country(skate_soup)

            all_points = get_all_points(skate_soup)
            places_t, scores_t = get_scores_places(all_points[0])
            places_c, scores_c = get_scores_places(all_points[1])
            if len(names) < 9:
                NO = [None for i in range(9 - len(names))]
                data.loc[data.shape[0]] = [name, country, year, event, gender, program, base_value] + places_t + NO + scores_t + NO + places_c + NO + scores_c + NO + names + NO + countries + NO
            elif len(names) == 9:
                data.loc[data.shape[0]] = [name, country, year, event, gender, program, base_value] + places_t + scores_t + places_c + scores_c + names + countries

            #data.loc[data.shape[0]] = [name, country, year, event, gender, program, base_value] + places_t + scores_t + places_c + scores_c + names + countries

    return data


# ----------- 6. Переименуем столбцы  ---------------------------------------


def rename_columns(data):
    columns_names = ['athlete_name', 'athlete_country', 'year', 'stage', 'sex', 'program', 'base_score']
    spis = [['judge_' + str(i + 1) + '_' + col  for i in range(9)] for col in ['tech_place', 'tech_score', 'component_place', 'component_score', 'name', 'country']]
    for row in spis:
        columns_names += row
    data = data.rename(columns = {i : columns_names[i] for i in range(data.shape[1])})
    data['event'] = data['year'].astype(str) + '_' + data['stage'].apply(lambda x: x.replace(' ', '_'))
    data['event_country'] = data['event'].apply(lambda x: country[x])

    #  1. Добавляю столбец home_advantage
    data['home_advantage'] = data['athlete_country'] == data['event_country']
    data['home_advantage'] = data['home_advantage'].astype(int)

    #  2. Меняю формат столбцов на более удобный
    data['year'] = data['year'].astype(int)
    decode = {'Mens' : 1, 'Womens' : 0, 'Pairs' : 2, 'Dance' : 3 , 'FreeSkate' : 1, 'ShortProgram' : 0, 'RhythmDance' : 0, 'FreeDance' : 1}
    data['type'] = data['sex'].apply(lambda x: decode.get(x, -1))
    data['program_numeric'] = data['program'].apply(lambda x: decode.get(x, -1))

    #  3. У пар и танцоров возьмем только фамилии
    data['athlete_name_2'] = data['athlete_name'].copy()
    data.loc[(data['type'] == 2) | (data['type'] == 3), 'athlete_name_2'] = data.loc[(data['type'] == 2) | (data['type'] == 3), 'athlete_name_2'].apply(lambda x: ' / '.join([name.split()[-1] for name in x.split(' / ')]))
    data = data.drop(columns = ['athlete_name'])
    data = data.rename(columns = {'athlete_name_2' : 'athlete_name'})

    #  4. У некоторых заменим имена на правильные
    to_check = pd.read_excel('check.xlsx')
    mapping = to_check.loc[to_check['true_name'].notna(), ['athlete_name', 'true_name']]
    replace_dict = dict(zip(mapping['athlete_name'], mapping['true_name']))
    data['athlete_name'] = data['athlete_name'].replace(replace_dict)

    return data

# ----------- 7. Обёртка --------------------------------------------------

@app.command()
def main(
    output_path: Path = PROCESSED_DATA_DIR / "full_22_23_24.xlsx",
    check_path: Optional[Path] = RAW_DATA_DIR / "check.xlsx",
):
    """
    Parse ISU/SkatingScores pages into a judge-level dataset.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    urls = get_urls(BASE_URL, EVENTS, GENDERS, PROGRAMS)
    logger.info(f"Will parse {len(urls)} pages.")

    df = build_dataset(urls)  # вместо get_dataset(data, urls)
    df = rename_columns(df, check_path=check_path if check_path.exists() else None)

    df.to_excel(output_path, index=False)
    logger.success(f"Saved: {output_path}")


if __name__ == "__main__":
    app()
