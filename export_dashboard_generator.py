import json
from pathlib import Path

import numpy as np
import pandas as pd

# 분석 기준 및 스코어링을 설정값으로 분리해 재사용성을 높인다.
# MoM Top3는 "품목" 레벨에서 항상 충분한 비교 대상이 나오도록 기본값을 item 으로 둔다.
MOM_LEVEL: str = "item"  # "item_country" 또는 "item"
HORIZON_MONTHS_RECENT: int = 6   # H2 가정 시 최근 몇 개월을 "최근"으로 볼 것인지

# H2 추천을 특정 제품군으로 제한하고 싶을 때 사용
FOCUS_ITEMS: list[str] | None = [
    # 예시: 의료기기, 태양광 패널, 배터리만 보고 싶을 때
    # "의료기기",
    # "태양광 패널",
    # "배터리",
]

# 국가를 거시 지역(Region)으로 맵핑해, 지역별 전략 가중을 다르게 줄 수 있도록 한다.
COUNTRY_TO_REGION: dict[str, str] = {
    # 아시아
    "일본": "Asia",
    "중국": "Asia",
    "한국": "Asia",
    "베트남": "Asia",
    "태국": "Asia",
    "인도네시아": "Asia",
    "인도": "Asia",
    "싱가포르": "Asia",
    "말레이시아": "Asia",
    "사우디아라비아": "Asia",
    "UAE": "Asia",
    "터키": "Asia",

    # 유럽
    "독일": "Europe",
    "프랑스": "Europe",
    "영국": "Europe",
    "이탈리아": "Europe",
    "스페인": "Europe",
    "네덜란드": "Europe",
    "폴란드": "Europe",

    # 미주
    "미국": "Americas",
    "캐나다": "Americas",
    "멕시코": "Americas",
    "브라질": "Americas",
    "아르헨티나": "Americas",

    # 기타/중동·아프리카 등
    "남아프리카": "MEA",
}

# Region 별 전략적 우선순위를 점수에 반영하기 위한 가중치
REGION_WEIGHTS: dict[str, float] = {
    "Asia": 1.10,      # 아시아 시장을 조금 더 공격적으로 보고 싶을 때
    "Europe": 1.00,
    "Americas": 1.00,
    "MEA": 1.05,
}


def load_data(csv_path: Path) -> pd.DataFrame:
    """CSV를 로드하고 기본 전처리를 수행한다."""
    # 인코딩 추론 실패에 대비해 순차적으로 시도
    encodings = ["utf-8-sig", "cp949", "utf-8"]
    last_err: Exception | None = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, skiprows=[0], encoding=enc)
            break
        except Exception as e:  # noqa: PERF203
            last_err = e
            df = None  # type: ignore[assignment]
    if df is None:
        raise RuntimeError(f"CSV 로드 실패: {last_err}")  # type: ignore[arg-type]

    # 컬럼 이름 표준화
    df = df.rename(
        columns={
            "연월": "month",
            "품목": "item",
            "수출국": "country",
            "수량(EA)": "quantity",
            "단가(USD)": "unit_price",
            "금액(USD)": "amount",
            "전월비(%)": "mom_pct",
        }
    )

    # 날짜/숫자형 변환
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    numeric_cols = ["quantity", "unit_price", "amount", "mom_pct"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .replace({"": np.nan})
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 결측값 처리: 수량/금액 결측은 0, 전월비는 유지
    df["quantity"] = df["quantity"].fillna(0)
    df["amount"] = df["amount"].fillna(0)

    return df


def compute_mom_top3(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """마지막 월 기준 전월 대비 Top3 변동을 계산한다.

    기본 규칙:
    - 마지막 월과 바로 직전 월을 먼저 비교한다.
    - 만약 공통 품목(또는 품목+국가 조합)이 전혀 없으면,
      → 더 이전 월들(예: 2026-01, 2025-12 ...)을 순차적으로 탐색해
        공통 조합이 나오는 첫 번째 월을 사용한다.
    - 비교 가능한 월이 전혀 없으면 빈 결과를 반환한다.
    """
    # 사용할 기준 월 목록(월 단위로 정규화)
    months_unique = sorted(df["month"].dt.to_period("M").unique())
    if not months_unique:
        empty = pd.DataFrame(
            columns=[
                "item_key",
                "amount_cur",
                "amount_prev",
                "abs_change",
                "change",
                "change_pct",
            ]
        )
        raise ValueError("월 정보가 없습니다.")

    max_period = months_unique[-1]
    max_month = max_period.to_timestamp()

    # 집계 기준: 설정값에 따라 품목 단위 또는 품목+국가 조합 단위로 계산
    df = df.copy()
    if MOM_LEVEL == "item":
        df["item_key"] = df["item"]
        group_cols: list[str] = ["month", "item_key"]
    else:
        df["item_key"] = df["item"] + " / " + df["country"]
        group_cols = ["month", "item_key"]

    monthly_item = df.groupby(group_cols, as_index=False)["amount"].sum()

    # 마지막 월 데이터
    cur_m = monthly_item[monthly_item["month"].dt.to_period("M") == max_period]
    if cur_m.empty:
        empty = pd.DataFrame(
            columns=[
                "item_key",
                "amount_cur",
                "amount_prev",
                "abs_change",
                "change",
                "change_pct",
            ]
        )
        # 이전 월이 있으면 그중 가장 최근 월을 prev_month 로 노출용 세팅
        prev_candidates = [p for p in months_unique if p < max_period]
        prev_month = (prev_candidates[-1] if prev_candidates else max_period - 1).to_timestamp()
        return empty, max_month, prev_month

    # 직전 월부터 과거로 내려가며 공통 item_key 가 생기는 첫 월을 탐색
    prev_candidates = [p for p in months_unique if p < max_period]
    chosen_prev_period: pd.Period | None = None
    merged: pd.DataFrame | None = None

    for cand in reversed(prev_candidates):
        prev_m = monthly_item[monthly_item["month"].dt.to_period("M") == cand]
        if prev_m.empty:
            continue
        m = cur_m.merge(prev_m, on="item_key", how="inner", suffixes=("_cur", "_prev"))
        if not m.empty:
            chosen_prev_period = cand
            merged = m
            break

    if chosen_prev_period is None or merged is None:
        # 비교 가능한 조합이 전혀 없을 때
        empty = pd.DataFrame(
            columns=[
                "item_key",
                "amount_cur",
                "amount_prev",
                "abs_change",
                "change",
                "change_pct",
            ]
        )
        # 시각용으로는 가장 최근 과거 월을 prev_month 로 사용
        prev_month = (prev_candidates[-1] if prev_candidates else max_period - 1).to_timestamp()
        return empty, max_month, prev_month

    prev_month = chosen_prev_period.to_timestamp()
    merged["abs_change"] = (merged["amount_cur"] - merged["amount_prev"]).abs()
    merged["change"] = merged["amount_cur"] - merged["amount_prev"]
    merged["change_pct"] = np.where(
        merged["amount_prev"] != 0,
        merged["change"] / merged["amount_prev"] * 100,
        np.nan,
    )

    top3 = merged.sort_values("abs_change", ascending=False).head(3).reset_index(drop=True)
    return top3, max_month, prev_month


def analyze_country_trends(df: pd.DataFrame) -> pd.DataFrame:
    """국가별 월간 합계와 추세(성장/감소/안정)를 분석한다."""
    monthly_country = (
        df.groupby(["month", "country"], as_index=False)["amount"]
        .sum()
        .sort_values(["country", "month"])
    )

    # 최근 성장률 및 장기 추세 계산
    trend_rows: list[dict[str, object]] = []
    for country, grp in monthly_country.groupby("country"):
        grp = grp.sort_values("month")
        values = grp["amount"].to_numpy()

        # 선형 회귀 기울기로 장기 추세 판단
        x = np.arange(len(values), dtype=float)
        if len(values) > 1 and not np.all(values == values[0]):
            slope, _ = np.polyfit(x, values, 1)
        else:
            slope = 0.0

        # 최근 3개월 성장률
        recent = grp.tail(3)
        if len(recent) >= 2:
            first_val = float(recent["amount"].iloc[0])
            last_val = float(recent["amount"].iloc[-1])
            recent_growth = (last_val - first_val) / first_val * 100 if first_val != 0 else 0.0
        else:
            recent_growth = 0.0

        # 변동성(표준편차/평균)
        mean_val = float(values.mean()) if len(values) > 0 else 0.0
        std_val = float(values.std()) if len(values) > 1 else 0.0
        volatility = std_val / mean_val * 100 if mean_val != 0 else 0.0

        # 최종 추세 라벨
        slope_norm = slope / mean_val * 100 if mean_val != 0 else 0.0
        if slope_norm > 5 and recent_growth > 0:
            trend = "성장"
        elif slope_norm < -5 and recent_growth < 0:
            trend = "감소"
        else:
            trend = "안정"

        trend_rows.append(
            {
                "country": country,
                "total_amount": float(values.sum()),
                "recent_growth": float(recent_growth),
                "volatility": float(volatility),
                "trend": trend,
            }
        )

    trend_df = pd.DataFrame(trend_rows)
    return monthly_country, trend_df


def recommend_h2_markets(
    df: pd.DataFrame,
    monthly_country: pd.DataFrame,
    trend_df: pd.DataFrame,
) -> pd.DataFrame:
    """H2 집중 추천 국가를 2~3개 선정한다."""
    # 1) 제품군 필터링: FOCUS_ITEMS 가 설정되어 있으면 해당 품목만 기반으로 H2 추천을 계산
    if FOCUS_ITEMS:
        focus_df = df[df["item"].isin(FOCUS_ITEMS)].copy()
        if focus_df.empty:
            # 필터 결과가 비면 전체 데이터 기반으로 폴백
            focus_df = df.copy()
    else:
        focus_df = df.copy()

    focus_monthly_country = (
        focus_df.groupby(["month", "country"], as_index=False)["amount"]
        .sum()
        .sort_values(["country", "month"])
    )

    max_month = focus_monthly_country["month"].max()
    min_month = focus_monthly_country["month"].min()

    # H2 가정: 데이터 마지막 월을 기준으로 최근 HORIZON_MONTHS_RECENT개월을 "최근"으로 정의
    if len(focus_monthly_country["month"].unique()) > HORIZON_MONTHS_RECENT:
        cutoff = max_month - pd.DateOffset(months=HORIZON_MONTHS_RECENT - 1)
    else:
        # 데이터가 짧으면 전체 기간을 사용
        cutoff = min_month

    recent = focus_monthly_country[focus_monthly_country["month"] >= cutoff]
    recent_totals = recent.groupby("country", as_index=False)["amount"].sum()
    recent_totals = recent_totals.rename(columns={"amount": "recent_amount"})

    merged = trend_df.merge(recent_totals, on="country", how="left")
    merged["recent_amount"] = merged["recent_amount"].fillna(0.0)

    # 2) 규모 점수: 최근 매출을 0~1로 정규화
    max_recent = merged["recent_amount"].max()
    if max_recent > 0:
        merged["score_amount"] = merged["recent_amount"] / max_recent
    else:
        merged["score_amount"] = 0.0

    # 3) 성장률 점수: -50%~+50%를 0~1 구간에 매핑 (바깥쪽은 클리핑)
    growth = merged["recent_growth"].clip(-50.0, 50.0)
    merged["score_growth"] = (growth + 50.0) / 100.0

    # 4) 안정성 점수: 변동성이 0%일 때 1, 100% 이상이면 0에 가깝도록 클리핑
    vol = merged["volatility"].clip(0.0, 100.0)
    merged["score_stability"] = 1.0 - (vol / 100.0)

    # 5) 장기 추세 보너스: 성장/안정/감소에 따라 가중 보정
    trend_bonus = []
    for trend in merged["trend"]:
        if trend == "성장":
            trend_bonus.append(0.08)
        elif trend == "안정":
            trend_bonus.append(0.04)
        else:  # "감소"
            trend_bonus.append(-0.06)
    merged["score_trend_bonus"] = trend_bonus

    # 6) 기본 스코어: 규모(0.45) + 성장(0.30) + 안정성(0.20) + 추세 보너스(±0.08)
    merged["score"] = (
        0.45 * merged["score_amount"]
        + 0.30 * merged["score_growth"]
        + 0.20 * merged["score_stability"]
        + merged["score_trend_bonus"]
    )

    # 7) 국가군(Region) 가중치 적용: 아시아·유럽·미주·MEA 등 전략 방향에 맞는 보정
    region_weight_list: list[float] = []
    for country in merged["country"]:
        region = COUNTRY_TO_REGION.get(str(country), "Others")
        weight = REGION_WEIGHTS.get(region, 1.0)
        region_weight_list.append(weight)
    merged["region_weight"] = region_weight_list
    merged["score_region_adj"] = merged["score"] * merged["region_weight"]

    # 성장 혹은 안정적인 국가를 우선 대상으로 삼되, 후보가 부족할 경우 전체 고려
    candidate = merged[merged["trend"].isin(["성장", "안정"])].copy()
    if candidate.empty or candidate.shape[0] < 2:
        candidate = merged.copy()

    rec = candidate.sort_values("score_region_adj", ascending=False).head(3).reset_index(drop=True)

    # 방어 로직: 추천 결과가 1~2개 이하로 나오면, 나머지는 규모 기준 상위 국가로 채운다.
    if rec.shape[0] < 3:
        needed = 3 - rec.shape[0]
        excluded = set(rec["country"].tolist())
        filler = (
            merged[~merged["country"].isin(excluded)]
            .sort_values("recent_amount", ascending=False)
            .head(needed)
        )
        if not filler.empty:
            rec = pd.concat([rec, filler], ignore_index=True)

    return rec


def build_html(
    df: pd.DataFrame,
    mom_top3: pd.DataFrame,
    monthly_country: pd.DataFrame,
    country_trends: pd.DataFrame,
    h2_rec: pd.DataFrame,
    max_month: pd.Timestamp,
    prev_month: pd.Timestamp,
) -> str:
    """대시보드용 HTML 문자열을 생성한다."""
    # 기간 요약
    min_month = df["month"].min()
    max_month_global = df["month"].max()
    date_range_text = f"{min_month.strftime('%Y-%m')} ~ {max_month_global.strftime('%Y-%m')}"

    # Chart.js용 데이터 직렬화
    # 1) MoM Top3 바차트 데이터
    mom_labels = mom_top3["item_key"].tolist()
    mom_values = mom_top3["change"].tolist()
    mom_change_pct = mom_top3["change_pct"].tolist()

    # 2) 국가별 시계열 차트 (상위 8개 국가)
    country_totals = (
        monthly_country.groupby("country", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
    )
    top_countries = country_totals["country"].head(8).tolist()
    ts = monthly_country[monthly_country["country"].isin(top_countries)]
    ts_pivot = ts.pivot(index="month", columns="country", values="amount").fillna(0.0)
    ts_labels = [d.strftime("%Y-%m") for d in ts_pivot.index]
    ts_datasets = []
    color_palette = [
        "#4F46E5",
        "#10B981",
        "#F97316",
        "#EF4444",
        "#06B6D4",
        "#8B5CF6",
        "#EC4899",
        "#22C55E",
    ]
    for idx, country in enumerate(ts_pivot.columns):
        ts_datasets.append(
            {
                "label": country,
                "data": ts_pivot[country].round(0).tolist(),
                "borderColor": color_palette[idx % len(color_palette)],
                "backgroundColor": color_palette[idx % len(color_palette)],
                "tension": 0.25,
                "fill": False,
            }
        )

    # 3) H2 추천 국가 데이터 (Region 레이블 및 가중치 포함)
    rec_cards = []
    for _, row in h2_rec.iterrows():
        country = str(row["country"])
        trend = str(row["trend"])
        region = COUNTRY_TO_REGION.get(country, "Others")
        region_weight = float(row.get("region_weight", 1.0))
        score_adj = float(row.get("score_region_adj", row.get("score", 0.0)))
        reasons = [
            f"최근 6개월 수출 규모: 약 {row['recent_amount'] / 1_000_000:.2f}M USD",
            f"최근 성장률: {row['recent_growth']:.1f}%",
            f"변동성(표준편차/평균): {row['volatility']:.1f}%",
            f"장기 추세: {trend}",
        ]
        rec_cards.append(
            {
                "country": country,
                "trend": trend,
                "region": region,
                "region_weight": region_weight,
                "score_adj": score_adj,
                "reasons": reasons,
            }
        )

    mom_summary_items = []
    for _, row in mom_top3.iterrows():
        direction = "상승" if row["change"] >= 0 else "감소"
        mom_summary_items.append(
            {
                "item": row["item_key"],
                "direction": direction,
                "abs_change": float(row["abs_change"]),
                "change_pct": float(row["change_pct"]) if not pd.isna(row["change_pct"]) else None,
            }
        )

    # MoM Top3 요약 텍스트용 안전 처리
    if not mom_top3.empty:
        top_abs_change = f"{mom_top3['abs_change'].iloc[0]:,.0f} USD"
        top_item_label = str(mom_top3["item_key"].iloc[0])
        top_change_val = float(mom_top3["change"].iloc[0])
        if not pd.isna(mom_top3["change_pct"].iloc[0]):
            top_change_str = f"{top_change_val:+,.0f} USD ({mom_top3['change_pct'].iloc[0]:.1f}%)"
        else:
            top_change_str = f"{top_change_val:+,.0f} USD"
    else:
        top_abs_change = "데이터 없음"
        top_item_label = "마지막 월과 전월 간 비교 가능한 품목이 없습니다."
        top_change_str = ""

    # HTML 템플릿
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>월별 수출 실적 데이터 대시보드</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {{
      --bg: #0f172a;
      --bg-elevated: #020617;
      --card: #020617;
      --card-soft: #02081f;
      --accent: #4f46e5;
      --accent-soft: rgba(79,70,229,0.15);
      --accent-strong: #6366f1;
      --text: #e5e7eb;
      --text-soft: #9ca3af;
      --border-subtle: rgba(148, 163, 184, 0.3);
      --danger: #f97373;
      --success: #22c55e;
      --shadow-soft: 0 22px 45px rgba(15,23,42,0.85);
      --radius-xl: 24px;
      --radius-lg: 18px;
      --radius-pill: 999px;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: "Pretendard", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top left, #1e293b 0, #020617 42%, #020617 100%);
      color: var(--text);
      min-height: 100vh;
      padding: 32px clamp(16px, 4vw, 40px);
      display: flex;
      justify-content: center;
      align-items: stretch;
    }}

    .shell {{
      max-width: 1320px;
      width: 100%;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }}

    .header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: center;
      padding: 18px 22px;
      border-radius: var(--radius-xl);
      border: 1px solid rgba(148, 163, 184, 0.25);
      background: radial-gradient(circle at top left, rgba(148,163,184,0.18), rgba(15,23,42,0.96));
      box-shadow: 0 24px 80px rgba(15,23,42,0.8);
      backdrop-filter: blur(18px);
    }}

    .header-main {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}

    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 12px;
      border-radius: var(--radius-pill);
      border: 1px solid rgba(148, 163, 184, 0.35);
      background: radial-gradient(circle at top left, rgba(79,70,229,0.5), transparent);
      font-size: 11px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--text-soft);
    }}

    .pulse-dot {{
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: #4ade80;
      box-shadow: 0 0 0 3px rgba(74, 222, 128, 0.25);
    }}

    .title {{
      font-size: clamp(22px, 2.1vw, 26px);
      font-weight: 600;
      letter-spacing: -0.03em;
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .title-emphasis {{
      padding: 4px 10px;
      border-radius: var(--radius-pill);
      background: rgba(15,118,110,0.16);
      border: 1px solid rgba(45,212,191,0.4);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #a5f3fc;
    }}

    .subtitle {{
      font-size: 13px;
      color: var(--text-soft);
    }}

    .header-meta {{
      display: flex;
      gap: 16px;
      align-items: stretch;
      flex-wrap: wrap;
    }}

    .meta-pill {{
      min-width: 160px;
      padding: 10px 14px;
      border-radius: var(--radius-lg);
      border: 1px solid rgba(148, 163, 184, 0.4);
      background: linear-gradient(135deg, rgba(15,23,42,0.4), rgba(15,23,42,0.1));
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}

    .meta-label {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: var(--text-soft);
    }}

    .meta-value {{
      font-size: 13px;
      font-weight: 500;
    }}

    .meta-pill.accent {{
      border-color: rgba(129, 140, 248, 0.8);
      background: radial-gradient(circle at top left, rgba(79,70,229,0.5), rgba(15,23,42,0.9));
      box-shadow: 0 18px 45px rgba(79,70,229,0.55);
    }}

    .layout-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      align-items: stretch;
    }}

    .card-h2 {{
      grid-column: 1 / -1;
      min-height: 0;  /* 여유 높이를 모두 제거하고 컨텐츠에만 맞춤 */
      padding-top: 18px;
      padding-bottom: 12px;
      background: radial-gradient(circle at top left, rgba(56,189,248,0.3), rgba(15,23,42,1));
      border-color: rgba(56,189,248,0.85);
      box-shadow: 0 24px 60px rgba(8,47,73,0.9);
    }}

    .card {{
      position: relative;
      border-radius: var(--radius-xl);
      background: radial-gradient(circle at top left, rgba(30,64,175,0.2), rgba(15,23,42,1));
      border: 1px solid rgba(148, 163, 184, 0.35);
      padding: 18px 18px 16px;
      box-shadow: var(--shadow-soft);
      overflow: hidden;
    }}

    .card::before {{
      content: "";
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at top right, rgba(79,70,229,0.35), transparent 55%);
      opacity: 0.5;
      pointer-events: none;
    }}

    .card-inner {{
      position: relative;
      z-index: 1;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}

    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
    }}

    .card-title {{
      font-size: 14px;
      font-weight: 600;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      color: #e5e7eb;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .chip {{
      font-size: 11px;
      padding: 3px 8px;
      border-radius: var(--radius-pill);
      border: 1px solid rgba(148, 163, 184, 0.5);
      color: var(--text-soft);
      background: rgba(15,23,42,0.7);
    }}

    .card-description {{
      font-size: 12px;
      color: var(--text-soft);
    }}

    .mom-highlight {{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 12px;
      border-radius: 16px;
      padding: 12px 14px;
      background: linear-gradient(135deg, rgba(5,46,22,0.8), rgba(15,23,42,0.9));
      border: 1px solid rgba(34,197,94,0.65);
    }}

    .mom-main-value {{
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}

    .mom-label {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: #bbf7d0;
    }}

    .mom-value {{
      font-size: 22px;
      font-weight: 600;
      letter-spacing: -0.03em;
      color: #ecfdf5;
    }}

    .mom-period {{
      font-size: 11px;
      color: #a7f3d0;
    }}

    .mom-chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 9px;
      border-radius: var(--radius-pill);
      background: rgba(22,163,74,0.18);
      border: 1px solid rgba(74,222,128,0.75);
      font-size: 11px;
      color: #bbf7d0;
    }}

    .mom-chip.down {{
      background: rgba(185,28,28,0.18);
      border-color: rgba(248,113,113,0.8);
      color: #fee2e2;
    }}

    .arrow-up {{
      border-left: 5px solid transparent;
      border-right: 5px solid transparent;
      border-bottom: 7px solid #22c55e;
      width: 0;
      height: 0;
    }}

    .arrow-down {{
      border-left: 5px solid transparent;
      border-right: 5px solid transparent;
      border-top: 7px solid #f97373;
      width: 0;
      height: 0;
    }}

    .mom-list {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 8px;
    }}

    .mom-item {{
      border-radius: 14px;
      padding: 8px 9px;
      background: rgba(15,23,42,0.85);
      border: 1px solid rgba(55,65,81,0.9);
      display: flex;
      flex-direction: column;
      gap: 4px;
      transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
      cursor: default;
    }}

    .mom-item:hover {{
      transform: translateY(-2px) translateZ(0);
      box-shadow: 0 16px 35px rgba(15,23,42,0.9);
      border-color: rgba(129,140,248,0.9);
    }}

    .mom-item-name {{
      font-size: 11px;
      color: var(--text-soft);
    }}

    .mom-item-value {{
      font-size: 12px;
      font-weight: 500;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
    }}

    .mom-item-dir.up {{
      color: var(--success);
    }}

    .mom-item-dir.down {{
      color: var(--danger);
    }}

    .chart-container {{
      position: relative;
      width: 100%;
      height: 260px;
    }}

    .chart-container-sm {{
      height: 200px;
    }}

    .legend-inline {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      font-size: 11px;
      color: var(--text-soft);
    }}

    .legend-dot {{
      width: 9px;
      height: 9px;
      border-radius: 999px;
      margin-right: 4px;
    }}

    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 4px 8px;
      border-radius: 999px;
      background: rgba(15,23,42,0.85);
      border: 1px solid rgba(55,65,81,0.9);
      cursor: pointer;
      transition: background-color 0.15s ease, border-color 0.15s ease, transform 0.12s ease;
    }}

    .legend-item.inactive {{
      opacity: 0.4;
      border-color: rgba(55,65,81,0.6);
      background: rgba(15,23,42,0.6);
    }}

    .legend-item:hover {{
      transform: translateY(-1px);
      border-color: rgba(129,140,248,0.8);
    }}

    .country-meta-list {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
    }}

    .country-meta-item {{
      font-size: 11px;
      color: var(--text-soft);
    }}

    .pill-trend {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 9px;
      border-radius: var(--radius-pill);
      font-size: 11px;
      border: 1px solid rgba(148, 163, 184, 0.6);
      background: rgba(15,23,42,0.7);
    }}

    .pill-trend.growth {{
      border-color: rgba(34,197,94,0.8);
      background: linear-gradient(135deg, rgba(22,163,74,0.35), rgba(15,23,42,0.95));
      color: #bbf7d0;
    }}

    .pill-trend.decline {{
      border-color: rgba(248,113,113,0.8);
      background: linear-gradient(135deg, rgba(185,28,28,0.3), rgba(15,23,42,0.95));
      color: #fee2e2;
    }}

    .pill-trend.stable {{
      border-color: rgba(148,163,184,0.75);
      background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(15,23,42,0.95));
      color: #e5e7eb;
    }}

    .rec-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}

    .rec-card {{
      border-radius: var(--radius-lg);
      background: radial-gradient(circle at top left, rgba(56,189,248,0.22), rgba(15,23,42,0.95));
      border: 1px solid rgba(56,189,248,0.8);
      padding: 12px 13px;
      display: flex;
      flex-direction: column;
      gap: 7px;
      box-shadow: 0 20px 35px rgba(8,47,73,0.7);
      transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
      cursor: default;
    }}

    .rec-card:hover {{
      transform: translateY(-3px) translateZ(0);
      box-shadow: 0 26px 50px rgba(8,47,73,0.9);
      border-color: rgba(56,189,248,1);
    }}

    .rec-country {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      font-size: 14px;
      font-weight: 600;
    }}

    .rec-meta-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 6px;
      font-size: 10px;
      color: var(--text-soft);
    }}

    .rec-region-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 3px 8px;
      border-radius: var(--radius-pill);
      border: 1px solid rgba(56,189,248,0.7);
      background: rgba(8,47,73,0.9);
      font-size: 10px;
      color: #e0f2fe;
    }}

    .rec-region-pill span:first-child {{
      font-weight: 500;
    }}

    .rec-score-tag {{
      padding: 2px 6px;
      border-radius: var(--radius-pill);
      background: rgba(15,23,42,0.9);
      border: 1px solid rgba(148,163,184,0.55);
      font-size: 10px;
      color: var(--text-soft);
    }}

    .rec-reasons {{
      font-size: 11px;
      color: #e5e7eb;
      padding-left: 10px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}

    .rec-reasons li {{
      list-style: none;
      position: relative;
      padding-left: 10px;
    }}

    .rec-reasons li::before {{
      content: "";
      position: absolute;
      left: 0;
      top: 50%;
      transform: translateY(-50%);
      width: 4px;
      height: 4px;
      border-radius: 999px;
      background: rgba(56,189,248,0.8);
    }}

    @media (max-width: 1080px) {{
      .layout-grid {{
        grid-template-columns: 1fr;
      }}
      .mom-list {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .rec-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}

    @media (max-width: 768px) {{
      body {{
        padding: 18px 12px;
      }}
      .header {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .header-meta {{
        width: 100%;
      }}
      .country-meta-list {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .rec-grid {{
        grid-template-columns: minmax(0, 1fr);
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="header">
      <div class="header-main">
        <div class="badge">
          <span class="pulse-dot"></span>
          <span>Export Performance • Live Snapshot</span>
        </div>
        <div class="title">
          월별 수출 실적 인사이트 대시보드
          <span class="title-emphasis">H2 전략 초점</span>
        </div>
        <p class="subtitle">
          월별·국가별 수출 실적을 기반으로 전월 대비 변동, 국가별 추세, 그리고 하반기 집중 공략 시장을 한 눈에 제공합니다.
        </p>
      </div>
      <div class="header-meta">
        <div class="meta-pill accent">
          <span class="meta-label">데이터 기간</span>
          <span class="meta-value">{date_range_text}</span>
        </div>
        <div class="meta-pill">
          <span class="meta-label">마지막 월</span>
          <span class="meta-value">{max_month.strftime('%Y-%m')}</span>
        </div>
        <div class="meta-pill">
          <span class="meta-label">거래 수</span>
          <span class="meta-value">{len(df):,} rows</span>
        </div>
      </div>
    </section>

    <section class="layout-grid">
      <article class="card">
          <div class="card-inner">
            <header class="card-header">
              <div>
                <h2 class="card-title">
                  MoM Top 3 변동 품목/국가
                  <span class="chip">Month over Month Δ</span>
                </h2>
                <p class="card-description">
                  마지막 월({max_month.strftime('%Y-%m')}) 기준 전월({prev_month.strftime('%Y-%m')}) 대비 금액 변동이 가장 큰 상위 3개 품목·국가 조합입니다.
                </p>
              </div>
            </header>

            <div class="mom-highlight">
              <div class="mom-main-value">
                <span class="mom-label">가장 큰 절대 변동</span>
                <div class="mom-value">
                  {top_abs_change}
                </div>
                <span class="mom-period">
                  {top_item_label}
                </span>
              </div>
              <div>
                {""
 if mom_top3.empty
 else f'''
                <div class="mom-chip {"down" if mom_top3['change'].iloc[0] < 0 else ""}">
                  <span class="{ 'arrow-down' if mom_top3['change'].iloc[0] < 0 else 'arrow-up' }"></span>
                  <span>
                    {top_change_str}
                  </span>
                </div>
                '''}
              </div>
            </div>

            <div class="chart-container chart-container-sm">
              <canvas id="momBarChart"></canvas>
            </div>

            <div class="mom-list" id="momList"></div>
          </div>
        </article>

      <article class="card">
          <div class="card-inner">
            <header class="card-header">
              <div>
                <h2 class="card-title">
                  국가별 월간 수출 추이
                  <span class="chip">Country Time Series</span>
                </h2>
                <p class="card-description">
                  상위 국가의 월별 수출 금액 변화를 한 번에 비교합니다. 라인 위에 마우스를 올려 국가별 세부 값을 확인할 수 있습니다.
                </p>
              </div>
            </header>

            <div class="chart-container">
              <canvas id="countryLineChart"></canvas>
            </div>

            <div class="legend-inline" id="countryLegend"></div>

            <div class="country-meta-list">
              <div class="country-meta-item">
                · 분석 국가 수: <strong>{country_trends.shape[0]}</strong>
              </div>
              <div class="country-meta-item">
                · 성장 국가 비중: <strong>{(country_trends['trend'] == '성장').mean() * 100:.1f}%</strong>
              </div>
              <div class="country-meta-item">
                · 감소/조정 국가는 포트폴리오 리밸런싱 후보로 볼 수 있습니다.
              </div>
            </div>
          </div>
        </article>

      <article class="card card-h2">
          <div class="card-inner">
            <header class="card-header">
              <div>
                <h2 class="card-title">
                  H2 집중 공략 추천 국가
                  <span class="chip">Focus Markets for H2</span>
                </h2>
                <p class="card-description">
                  최근 수개월의 실적 규모, 성장률, 변동성을 종합해 하반기(H2)에 전략적으로 집중할 만한 2~3개 주요 국가를 제안합니다.
                </p>
              </div>
            </header>

            <div class="rec-grid" id="recGrid"></div>
          </div>
        </article>
    </section>
  </main>

  <script>
    const momLabels = {json.dumps(mom_labels, ensure_ascii=False)};
    const momValues = {json.dumps(mom_values, ensure_ascii=False)};
    const momChangePct = {json.dumps(mom_change_pct, ensure_ascii=False)};
    const momSummary = {json.dumps(mom_summary_items, ensure_ascii=False)};

    const tsLabels = {json.dumps(ts_labels, ensure_ascii=False)};
    const tsDatasets = {json.dumps(ts_datasets, ensure_ascii=False)};

    const h2Recommendations = {json.dumps(rec_cards, ensure_ascii=False)};

    function formatUsd(value) {{
      return value.toLocaleString('en-US', {{ maximumFractionDigits: 0 }});
    }}

    function buildMomList() {{
      const container = document.getElementById('momList');
      container.innerHTML = '';
      momSummary.forEach((row) => {{
        const item = document.createElement('div');
        item.className = 'mom-item';
        const dirClass = row.direction === '상승' ? 'up' : 'down';
        const sign = row.direction === '상승' ? '+' : '';
        const pctText = row.change_pct !== null
          ? ' (' + row.change_pct.toFixed(1) + '%)'
          : '';
        item.innerHTML = `
          <div class="mom-item-name">${{row.item}}</div>
          <div class="mom-item-value">
            <span class="mom-item-dir ${{dirClass}}">
              ${{row.direction}}
            </span>
            <span>
              ${{sign}}${{formatUsd(row.abs_change)}} USD${{pctText}}
            </span>
          </div>
        `;
        container.appendChild(item);
      }});
    }}

    function buildRecCards() {{
      const grid = document.getElementById('recGrid');
      grid.innerHTML = '';
      h2Recommendations.forEach((rec) => {{
        const card = document.createElement('div');
        card.className = 'rec-card';
        let trendClass = 'stable';
        if (rec.trend === '성장') trendClass = 'growth';
        else if (rec.trend === '감소') trendClass = 'decline';

        const reasonsHtml = rec.reasons.map((r) => `<li>${{r}}</li>`).join('');
        const regionLabel = rec.region || 'N/A';
        const regionWeight = typeof rec.region_weight === 'number'
          ? '×' + rec.region_weight.toFixed(2)
          : '';
        const scoreText = typeof rec.score_adj === 'number'
          ? '종합 점수: ' + (rec.score_adj * 100).toFixed(1) + ' / 100'
          : '';

        card.innerHTML = `
          <div class="rec-country">
            <span>${{rec.country}}</span>
            <span class="pill-trend ${{trendClass}}">${{rec.trend}}</span>
          </div>
          <div class="rec-meta-row">
            <div class="rec-region-pill">
              <span>${{regionLabel}}</span>
              <span>${{regionWeight}}</span>
            </div>
            <span class="rec-score-tag">${{scoreText}}</span>
          </div>
          <ul class="rec-reasons">
            ${{reasonsHtml}}
          </ul>
        `;
        grid.appendChild(card);
      }});
    }}

    function createMomBarChart() {{
      const ctx = document.getElementById('momBarChart').getContext('2d');
      const gradient = ctx.createLinearGradient(0, 0, 0, 200);
      gradient.addColorStop(0, 'rgba(129,140,248,0.95)');
      gradient.addColorStop(1, 'rgba(129,140,248,0.15)');

      new Chart(ctx, {{
        type: 'bar',
        data: {{
          labels: momLabels,
          datasets: [{{
            data: momValues,
            backgroundColor: (ctx) => {{
              const value = ctx.raw || 0;
              return value >= 0 ? 'rgba(52,211,153,0.6)' : 'rgba(248,113,113,0.6)';
            }},
            borderRadius: 12,
            borderSkipped: false,
          }}],
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{
            legend: {{ display: false }},
            tooltip: {{
              backgroundColor: 'rgba(15,23,42,0.95)',
              borderColor: 'rgba(148,163,184,0.5)',
              borderWidth: 1,
              padding: 10,
              callbacks: {{
                label: (ctx) => {{
                  const value = ctx.raw || 0;
                  const pct = momChangePct[ctx.dataIndex];
                  const pctText = pct === null
                    ? ''
                    : ' (' + pct.toFixed(1) + '%)';
                  const sign = value >= 0 ? '+' : '';
                  return sign + formatUsd(value) + ' USD' + pctText;
                }},
              }},
            }},
          }},
          scales: {{
            x: {{
              grid: {{
                display: false,
              }},
              ticks: {{
                color: '#9ca3af',
                font: {{
                  size: 10,
                }},
              }},
            }},
            y: {{
              grid: {{
                color: 'rgba(55,65,81,0.8)',
                drawBorder: false,
              }},
              ticks: {{
                color: '#9ca3af',
                callback: (value) => {{
                  return formatUsd(value);
                }},
                font: {{
                  size: 10,
                }},
              }},
            }},
          }},
        }},
      }});
    }}

    function createCountryLineChart() {{
      const ctx = document.getElementById('countryLineChart').getContext('2d');
      const legend = document.getElementById('countryLegend');

      // 초기에는 모든 국가를 활성 상태로 두되, 범례 클릭으로 개별 토글
      const activeMap = {{}};
      tsDatasets.forEach((ds) => {{
        activeMap[ds.label] = true;
      }});

      const chart = new Chart(ctx, {{
        type: 'line',
        data: {{
          labels: tsLabels,
          datasets: tsDatasets.map((ds) => ({{
            ...ds,
            pointRadius: 2.5,
            pointHoverRadius: 4,
          }})),
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          interaction: {{
            mode: 'index',
            intersect: false,
          }},
          plugins: {{
            legend: {{ display: false }},
            tooltip: {{
              backgroundColor: 'rgba(15,23,42,0.95)',
              borderColor: 'rgba(148,163,184,0.6)',
              borderWidth: 1,
              padding: 10,
              callbacks: {{
                label: (ctx) => {{
                  const label = ctx.dataset.label || '';
                  const value = ctx.raw || 0;
                  return `${{label}}: ${{formatUsd(value)}} USD`;
                }},
              }},
            }},
          }},
          scales: {{
            x: {{
              grid: {{
                color: 'rgba(30,64,175,0.35)',
                drawBorder: false,
              }},
              ticks: {{
                color: '#9ca3af',
                maxRotation: 0,
                autoSkipPadding: 12,
                font: {{
                  size: 10,
                }},
              }},
            }},
            y: {{
              grid: {{
                color: 'rgba(55,65,81,0.85)',
                drawBorder: false,
              }},
              ticks: {{
                color: '#9ca3af',
                callback: (value) => formatUsd(value),
                font: {{
                  size: 10,
                }},
              }},
            }},
          }},
        }},
      }});

      legend.innerHTML = '';
      chart.data.datasets.forEach((ds, index) => {{
        const item = document.createElement('div');
        item.className = 'legend-item';
        const dot = document.createElement('span');
        dot.className = 'legend-dot';
        dot.style.backgroundColor = ds.borderColor;
        const label = document.createElement('span');
        label.textContent = ds.label;
        item.appendChild(dot);
        item.appendChild(label);
        legend.appendChild(item);

        // 클릭 시 해당 국가 라인 표시/숨김 토글
        item.addEventListener('click', () => {{
          const current = activeMap[ds.label];
          activeMap[ds.label] = !current;

          chart.isDatasetVisible(index)
            ? chart.hide(index)
            : chart.show(index);

          if (activeMap[ds.label]) {{
            item.classList.remove('inactive');
          }} else {{
            item.classList.add('inactive');
          }}
        }});
      }});
    }}

    document.addEventListener('DOMContentLoaded', () => {{
      buildMomList();
      buildRecCards();
      createMomBarChart();
      createCountryLineChart();
    }});
  </script>
</body>
</html>
"""
    return html


def main() -> None:
    """CSV를 분석하고 export_dashboard.html을 생성한다."""
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "예제2.CSV"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    df = load_data(csv_path)
    mom_top3, max_month, prev_month = compute_mom_top3(df)
    monthly_country, country_trends = analyze_country_trends(df)
    h2_rec = recommend_h2_markets(df, monthly_country, country_trends)

    html = build_html(
        df=df,
        mom_top3=mom_top3,
        monthly_country=monthly_country,
        country_trends=country_trends,
        h2_rec=h2_rec,
        max_month=max_month,
        prev_month=prev_month,
    )

    out_path = base_dir / "export_dashboard.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"대시보드가 생성되었습니다: {out_path}")


if __name__ == "__main__":
    main()

