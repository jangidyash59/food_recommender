# app.py — FINAL: Auto-select state with most txns for typed item
import streamlit as st
import pandas as pd
import glob, os
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Indian Food Recommender", page_icon="🍴", layout="centered")
st.title("🍴 Statewise Restaurant Food Recommender System")
st.caption("Restaurant-level FP-Growth with Top Picks & Per-Restaurant Recommendations")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    food_df = pd.read_csv("indian_food_use.csv")
    food_df.columns = food_df.columns.str.strip().str.lower()
    food_df["name_norm"] = food_df["name"].astype(str).str.strip().str.lower()
    food_df["state_norm"] = food_df["state"].astype(str).str.strip().str.lower().str.replace(" ", "_")

    # allowed prefixes (expandable)
    allowed_prefixes = {"west_bengal", "rajasthan", "gujarat", "maharashtra"}

    restaurant_files = sorted(glob.glob("*.csv"))
    exclude = {"indian_food_use.csv", "national_transactions.csv"}
    restaurant_files = [f for f in restaurant_files if os.path.basename(f) not in exclude]

    restaurants = {}
    for f in restaurant_files:
        key = os.path.splitext(os.path.basename(f))[0].strip()
        key_lower = key.lower()

        # match allowed prefix by startswith
        found_prefix = None
        for p in allowed_prefixes:
            if key_lower.startswith(p):
                found_prefix = p
                break
        if found_prefix is None:
            continue

        df = pd.read_csv(f)
        df.columns = df.columns.str.strip().str.lower()
        if "transaction_items" not in df.columns:
            continue
        df["transaction_items"] = df["transaction_items"].astype(str)
        restaurants[key_lower] = df.copy()

    return food_df, restaurants, allowed_prefixes

food_df, restaurants, allowed_prefixes = load_data()
st.info(f"✅ Loaded food metadata ({food_df.shape[0]} rows) and {len(restaurants)} restaurant datasets: {', '.join(restaurants.keys())}")

# -------------------- HELPERS --------------------
def parse_txn_items(t: str):
    return [x.split(" (")[0].strip().lower() for x in str(t).split(";") if x.strip()]

def get_item_count(df: pd.DataFrame, item: str):
    item = item.strip().lower()
    return sum(item in parse_txn_items(t) for t in df["transaction_items"].astype(str))

def top_items_in_restaurant(df: pd.DataFrame, top_n=10):
    counter = {}
    for t in df["transaction_items"].astype(str):
        for i in parse_txn_items(t):
            counter[i] = counter.get(i, 0) + 1
    return sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_n]

def run_fp_with_fallback(df: pd.DataFrame, supports=(0.1, 0.05, 0.02, 0.01, 0.005, 0.001)):
    txns = [parse_txn_items(t) for t in df["transaction_items"].astype(str)]
    if not txns:
        return pd.DataFrame()
    for s in supports:
        try:
            te = TransactionEncoder()
            te_ary = te.fit(txns).transform(txns)
            df_enc = pd.DataFrame(te_ary, columns=te.columns_)
            freq_items = fpgrowth(df_enc, min_support=s, use_colnames=True)
            if freq_items.empty:
                continue
            rules = association_rules(freq_items, metric="confidence", min_threshold=0.05)
            if not rules.empty:
                return rules
        except Exception:
            continue
    return pd.DataFrame()

def restaurants_by_state(state_prefix: str):
    return [r for r in restaurants.keys() if r.startswith(state_prefix)]

def get_available_prefixes(loaded_rest_keys, all_prefixes):
    found = set()
    for key in loaded_rest_keys:
        for p in all_prefixes:
            if key.startswith(p):
                found.add(p)
                break
    return sorted(list(found))

# -------------------- SESSION STATE KEYS --------------------
if "top2_for_query" not in st.session_state:
    st.session_state.top2_for_query = None
if "top2_query_dish" not in st.session_state:
    st.session_state.top2_query_dish = None
if "top2_state" not in st.session_state:
    st.session_state.top2_state = None

# -------------------- UI --------------------
st.header("🍛 Step 1 • Choose Your Action")
choice = st.radio("Select an option:", ["🍴 Food Recommendation", "⭐ Top Picks"])

# -------------------- FOOD RECOMMENDATION --------------------
# -------------------- FOOD RECOMMENDATION --------------------
if choice == "🍴 Food Recommendation":
    st.subheader("🍴 Food Recommendation (Restaurant-level)")

    # --- Dropdown list of all food items (editable + searchable) ---
    unique_dishes = sorted(food_df["name_norm"].dropna().unique().tolist())
    unique_dishes_display = [name.title() for name in unique_dishes]

    # Editable dropdown with auto-suggestions
    food_input = st.selectbox(
        "Enter or Select a Dish Name:",
        options=[""] + unique_dishes_display,
        index=0,
        help="Start typing to search for a dish (auto-suggestions enabled)"
    ).strip()

    diet_choice = st.selectbox("Filter by Diet Type:", ["Both", "Vegetarian", "Non Vegetarian"])

    # determine available state prefixes from loaded restaurant files
    available_state_prefixes = get_available_prefixes(restaurants.keys(), allowed_prefixes)
    if not available_state_prefixes:
        st.warning("No restaurant data files found. Please add CSVs with allowed prefixes.")
    else:
        state_display_map = {p: p.replace("_", " ").title() for p in available_state_prefixes}

        # --- AUTO-SELECT STATE BASED ON typed item ---
        default_index = 0
        if food_input:
            item = food_input.lower()
            state_counts = {}
            for prefix in available_state_prefixes:
                total = 0
                for rest in restaurants_by_state(prefix):
                    total += get_item_count(restaurants[rest], item)
                state_counts[prefix] = total

            if any(state_counts.values()):
                # State with max transactions
                max_prefix = max(state_counts, key=state_counts.get)
                default_index = available_state_prefixes.index(max_prefix)
                st.info(
                    f"✅ Auto-selected state: **{state_display_map[max_prefix]}** — most transactions found for '{food_input.title()}'."
                )
            else:
                st.warning(f"No transactions found for '{food_input.title()}' in any state.")

        # --- State dropdown (auto-selected) ---
        state_choice_prefix = st.selectbox(
            "Select State:",
            available_state_prefixes,
            index=default_index,
            format_func=lambda x: state_display_map[x]
        )
        state_choice = state_display_map[state_choice_prefix]

        # --- Find Top Restaurants button ---
        if st.button("🔍 Find Top Restaurants"):
            state_restaurants = restaurants_by_state(state_choice_prefix)
            rest_counts = [(r, get_item_count(restaurants[r], food_input)) for r in state_restaurants]

            total_txns = sum(count for _, count in rest_counts)
            if total_txns == 0:
                st.warning(f"No transactions found for '{food_input.title()}' in {state_choice}.")
                st.session_state.top2_for_query = None
                st.session_state.top2_query_dish = None
                st.session_state.top2_state = None
            else:
                sorted_counts = sorted(rest_counts, key=lambda x: x[1], reverse=True)
                top2_names = [r[0] for r in sorted_counts[:2]]
                unified_table = []
                for name, count in sorted_counts:
                    star = "⭐" if name in top2_names else ""
                    unified_table.append({"Restaurant": f"{name.title()} {star}".strip(), "Transactions": count})

                st.success(f"📊 Item count & 🏆 Top Restaurants for '{food_input.title()}' in {state_choice}:")
                st.dataframe(pd.DataFrame(unified_table))

                top2 = [r for r in sorted_counts[:2] if r[1] > 0]
                st.session_state.top2_for_query = top2
                st.session_state.top2_query_dish = food_input.lower()
                st.session_state.top2_state = state_choice_prefix

        # --- Show recommendations for top 2 ---
        if st.session_state.top2_for_query:
            top2 = st.session_state.top2_for_query
            choice_list = [r[0] for r in top2]
            restaurant_choice = st.selectbox("Select a Restaurant for Detailed Recommendations:", choice_list)

            if restaurant_choice and st.session_state.top2_query_dish:
                rdf = restaurants[restaurant_choice]

                # 🔁 Run FP-Growth with dynamic support adjustment
                supports = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
                rules = pd.DataFrame()
                for s in supports:
                    rules = run_fp_with_fallback(rdf, supports=(s,))
                    if not rules.empty:
                        break

                if rules.empty:
                    st.warning(f"No frequent combinations found for '{food_input.title()}' in {restaurant_choice} even at minimum support.")
                else:
                    item = st.session_state.top2_query_dish
                    mask = rules.apply(
                        lambda row: item in [i.lower() for i in list(row["antecedents"])]
                                    or item in [i.lower() for i in list(row["consequents"])],
                        axis=1
                    )
                    matched = rules[mask]

                    # If no direct combinations found, lower support dynamically and retry
                    if matched.empty:
                        for s in [0.001, 0.0005, 0.0001]:
                            rules = run_fp_with_fallback(rdf, supports=(s,))
                            if not rules.empty:
                                mask = rules.apply(
                                    lambda row: item in [i.lower() for i in list(row["antecedents"])]
                                                or item in [i.lower() for i in list(row["consequents"])],
                                    axis=1
                                )
                                matched = rules[mask]
                                if not matched.empty:
                                    st.info(f"📉 Auto-adjusted support to {s} for better results.")
                                    break

                    if matched.empty:
                        st.warning(f"No frequent combinations found for '{item.title()}' in {restaurant_choice}.")
                    else:
                        state_prefix = st.session_state.top2_state
                        state_restaurants = restaurants_by_state(state_prefix)
                        top_rest3 = [i[0] for i in top_items_in_restaurant(rdf, top_n=3)]
                        state_top5 = set()
                        for r in state_restaurants:
                            state_top5.update([i[0] for i in top_items_in_restaurant(restaurants[r], top_n=5)])

                        suggestions = set()
                        for _, rrow in matched.iterrows():
                            for it in list(rrow["antecedents"]) + list(rrow["consequents"]):
                                if it.lower() != item:
                                    suggestions.add(it.lower())

                        rows = []
                        for s in suggestions:
                            cnt = get_item_count(rdf, s)
                            if cnt <= 0:
                                continue
                            meta = food_df[food_df["name_norm"] == s]
                            diet = meta["diet"].iloc[0].title() if not meta.empty else "Unknown"
                            origin = meta["state"].iloc[0].title() if not meta.empty else "Unknown"
                            marker = ""
                            if s in top_rest3:
                                marker += "🟣"
                            if s in state_top5:
                                marker += "🟡"
                            rows.append({
                                "Dish": f"{marker} {s.title()}",
                                "Diet": diet,
                                "State of Origin": origin,
                                "Transactions": cnt
                            })
                        if rows:
                            df_out = pd.DataFrame(rows).sort_values(by="Transactions", ascending=False).head(10)
                            if diet_choice.lower() != "both":
                                df_out = df_out[df_out["Diet"].str.lower() == diet_choice.lower()]
                                df_out = df_out.drop(columns=["Diet"], errors="ignore")
                            st.success(f"🍽️ Recommendations for '{item.title()}' in {restaurant_choice}:")
                            st.dataframe(df_out.reset_index(drop=True))
                        else:
                            st.warning(f"No related items found for '{item.title()}' in {restaurant_choice}.")

        # legend for Food Recommendation
        st.markdown("---")
        st.markdown("**Legend:**")
        st.markdown("- ⭐ **Gold** = One of the Top 2 restaurants for the chosen dish.")
        st.markdown("- 🟣 **Purple** = Top 3 items in the selected restaurant.")
        st.markdown("- 🟡 **Yellow** = Top 5 items across restaurants in the selected state.")

# -------------------- TOP PICKS SECTION (unchanged) --------------------
# -------------------- TOP PICKS SECTION (with auto support adjustment) --------------------
else:
    st.subheader("⭐ Top Picks by State / Restaurant")
    diet_choice = st.selectbox("Filter by Diet Type:", ["Both", "Vegetarian", "Non Vegetarian"])

    available_state_prefixes = get_available_prefixes(restaurants.keys(), allowed_prefixes)
    if not available_state_prefixes:
        st.warning("No restaurant data files found. Please check your CSV files.")
    else:
        state_display_map = {p: p.replace("_", " ").title() for p in available_state_prefixes}
        state_choice_prefix = st.selectbox("Select State:", available_state_prefixes, format_func=lambda x: state_display_map[x])
        state_choice = state_display_map[state_choice_prefix]

        state_restaurants = restaurants_by_state(state_choice_prefix)
        state_items = food_df[food_df["state_norm"] == state_choice_prefix]["name_norm"].tolist()

        total_counts = {}
        per_item_rest_counts = {}
        for rest in state_restaurants:
            df = restaurants[rest]
            for t in df["transaction_items"].astype(str):
                for i in parse_txn_items(t):
                    if i not in state_items:
                        continue
                    if diet_choice.lower() != "both":
                        d = food_df.loc[food_df["name_norm"] == i, "diet"]
                        if not d.empty and d.iloc[0].lower() != diet_choice.lower():
                            continue
                    total_counts[i] = total_counts.get(i, 0) + 1
                    per_item_rest_counts.setdefault(i, {})
                    per_item_rest_counts[i][rest] = per_item_rest_counts[i].get(rest, 0) + 1

        if not total_counts:
            st.warning(f"No state-origin items found in transactions for {state_choice} matching the diet filter.")
        else:
            best_rest_for_item = {i: max(r.items(), key=lambda x: x[1])[0] for i, r in per_item_rest_counts.items()}
            top_items = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            rows = []
            for rank, (item, total_cnt) in enumerate(top_items):
                meta = food_df[food_df["name_norm"] == item]
                diet = meta["diet"].iloc[0].title() if not meta.empty else "Unknown"
                rest = best_rest_for_item.get(item, "Unknown")
                marker = "🟡" if rank < 5 else ""
                rows.append({
                    "Dish": f"{marker} {item.title()}",
                    "Diet": diet,
                    "Restaurant": rest,
                    "Overall Transactions": total_cnt
                })
            df_top = pd.DataFrame(rows).sort_values(by="Overall Transactions", ascending=False).reset_index(drop=True)
            st.success(f"🔝 Top Picks in {state_choice}:")
            st.dataframe(df_top)

            chosen_item = st.selectbox("Select a Dish from Top Picks for Recommendations:", df_top["Dish"])
            raw_item = chosen_item.lstrip("🟡🟣 ").lower()
            default_rest = best_rest_for_item.get(raw_item, state_restaurants[0])
            try:
                default_index = state_restaurants.index(default_rest)
            except ValueError:
                default_index = 0
            restaurant_choice = st.selectbox("Select a Restaurant:", state_restaurants, index=default_index)

            if st.button("🍽️ Get Recommendations from Top Pick"):
                rdf = restaurants[restaurant_choice]

                # 🔁 Run FP-Growth with dynamic support adjustment
                supports = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
                rules = pd.DataFrame()
                used_support = None
                for s in supports:
                    rules = run_fp_with_fallback(rdf, supports=(s,))
                    if not rules.empty:
                        used_support = s
                        break

                if rules.empty:
                    st.warning(f"No frequent combinations found for '{raw_item.title()}' in {restaurant_choice} even at minimum support.")
                else:
                    item = raw_item
                    mask = rules.apply(
                        lambda row: item in [i.lower() for i in list(row["antecedents"])]
                                    or item in [i.lower() for i in list(row["consequents"])],
                        axis=1
                    )
                    matched = rules[mask]

                    # If none found, lower support automatically
                    if matched.empty:
                        for s in [0.001, 0.0005, 0.0001]:
                            rules = run_fp_with_fallback(rdf, supports=(s,))
                            if not rules.empty:
                                mask = rules.apply(
                                    lambda row: item in [i.lower() for i in list(row["antecedents"])]
                                                or item in [i.lower() for i in list(row["consequents"])],
                                    axis=1
                                )
                                matched = rules[mask]
                                if not matched.empty:
                                    st.info(f"📉 Auto-adjusted support to {s} for better results.")
                                    used_support = s
                                    break

                    if matched.empty:
                        st.warning(f"No frequent combinations found for '{item.title()}' in {restaurant_choice}.")
                    else:
                        top_rest3 = [i[0] for i in top_items_in_restaurant(rdf, top_n=3)]
                        state_top5 = set()
                        for r in state_restaurants:
                            state_top5.update([i[0] for i in top_items_in_restaurant(restaurants[r], top_n=5)])

                        suggestions = set()
                        for _, rrow in matched.iterrows():
                            for it in list(rrow["antecedents"]) + list(rrow["consequents"]):
                                if it.lower() != item:
                                    suggestions.add(it.lower())

                        rows = []
                        for s in suggestions:
                            cnt = get_item_count(rdf, s)
                            if cnt <= 0:
                                continue
                            meta = food_df[food_df["name_norm"] == s]
                            diet = meta["diet"].iloc[0].title() if not meta.empty else "Unknown"
                            origin = meta["state"].iloc[0].title() if not meta.empty else "Unknown"
                            marker = ""
                            if s in top_rest3:
                                marker += "🟣"
                            if s in state_top5:
                                marker += "🟡"
                            rows.append({
                                "Dish": f"{marker} {s.title()}",
                                "Diet": diet,
                                "State of Origin": origin,
                                "Transactions": cnt
                            })
                        if rows:
                            df_out = pd.DataFrame(rows).sort_values(by="Transactions", ascending=False).head(10)
                            if diet_choice.lower() != "both":
                                df_out = df_out[df_out["Diet"].str.lower() == diet_choice.lower()]
                                df_out = df_out.drop(columns=["Diet"], errors="ignore")
                            msg = f"🍽️ Recommendations for '{item.title()}' in {restaurant_choice}"
                            if used_support:
                                msg += f" (min_support={used_support})"
                            st.success(msg + ":")
                            st.dataframe(df_out.reset_index(drop=True))
                        else:
                            st.warning(f"No related items found for '{item.title()}' in {restaurant_choice}.")
