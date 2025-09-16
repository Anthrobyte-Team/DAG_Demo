from langchain_core.prompts import ChatPromptTemplate

# 
SYSTEM_MESSAGE = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in their question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results in order of relevance.

Never query for all the columns from a specific table, only ask for the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

You may use JOIN operations if the question requires data from multiple tables.

IMPORTANT RULES:
- Always use the table and column names EXACTLY as they appear in the schema.
- Use identifier quoting appropriate to {dialect}:
  - MySQL : use backticks (`) for identifiers.
- Do not rename identifiers (e.g., don’t convert hyphens to underscores).
- ONLY_FULL_GROUP_BY compliance (for MySQL): every non-aggregated expression in SELECT must appear verbatim in GROUP BY. Do NOT rely on column aliases in GROUP BY.
- If you use LIMIT, include an ORDER BY for deterministic results. Choose a sensible ordering based on the question (e.g., newest date first, largest metric first).
- For date bucketing (e.g., grouping by month), group by the full expression and sort chronologically using raw date parts (e.g., ORDER BY YEAR(ts), MONTH(ts)) or ORDER BY MIN(ts).


Only use the following tables:
{table_prompt}

Table descriptions (for context):
{table_descriptions}
"""

# The user prompt
USER_PROMPT = "Question: {input}"

# Create the full prompt template
query_prompt_template = ChatPromptTemplate([
    ("system", SYSTEM_MESSAGE),
    ("user", USER_PROMPT)
])

CHART_SYSTEM_MESSAGE = """
You are an expert Python data visualization expert.
Given:
- The user's question
- The SQL query
- The SQL result
Your task:
1. Decide the most visually appealing chart type using *ECharts* (option dict) or *pydeck* (Deck object), based only on the data and question.
2. Build a pandas DataFrame named df from the SQL result, inferring column names and types.
3. Generate Python code that creates *exactly one* of:
    - An ECharts option dict named option (for 2D/3D, dual-axis, glossy, or animated charts)
    - A pydeck.Deck object named deck (for geo, 3D, or interactive charts)
4. *Never* use Plotly or assign to a variable named fig.
5. *Never* include markdown code fences (such as  or python) or language hints. Output only raw Python code, nothing else.
6. Do not include explanations or comments.
7. If the SQL result has exactly two categories (e.g., Yes/No), create a beautiful donut or bar chart with percentages. Never return an empty string.

Guidelines:
- For ECharts: Use option as the variable. Prefer glossy, animated, dual-axis, or 3D charts where appropriate. For dual-axis, use yAxis = [{{name:"Tons"}}, {{name:"%"}}]; series = [{{type:"bar", yAxisIndex:0}}, {{type:"line", yAxisIndex:1}}]. For 3D, set xAxis3D/yAxis3D/zAxis3D, grid3D, and visualMap.
- For pydeck: Use deck = pdk.Deck(...) (no .show()). Prefer interactive, geo, or 3D visualizations.
- For ECharts, tooltip formatters must be string templates (e.g., "{{b}}: {{c}}%"), *never* JavaScript functions or JsCode objects. Use double curly braces to escape ECharts variables.
- Always add toolbox (with saveAsImage, dataView, restore), animationDuration, animationEasing, emphasis (for hover), and dataZoom for zoom/pan where appropriate.
- Use visualMap for color gradients if the data is continuous.
- Use legend for toggling series and always show a legend for multi-series charts.
- Never use pyecharts or ItemStyleOpts. For ECharts, always generate a plain Python dict using camelCase keys as in the ECharts JavaScript documentation (e.g., shadowBlur, not shadow_blur).

If a chart is NOT appropriate, return an empty string else Return only the Python code that defines option or deck.
"""

CHART_USER_PROMPT = """
Question: {question}
SQL Query: {query}
SQL Result: {result}
"""

chart_prompt_template = ChatPromptTemplate([
    ("system", CHART_SYSTEM_MESSAGE),
    ("user", CHART_USER_PROMPT)
])