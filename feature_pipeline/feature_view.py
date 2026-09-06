import os
import hopsworks

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# CONFIGURATION

FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 10  

FEATURE_VIEW_NAME = "karachi_aqi_feature_view"
FEATURE_VIEW_VERSION = 10  

LABEL_COLUMNS = [
    "target_aqi_24",
    "target_aqi_48",
    "target_aqi_72",
]

EXCLUDED_COLUMNS = [
    "target_change_24",
    "target_change_48",
    "target_change_72",
]


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

# 1. CONNECT TO HOPSWORKS

print_section("CONNECTING TO HOPSWORKS")

api_key = os.environ.get("HOPSWORKS_API_KEY")
project_name = os.environ.get("HOPSWORKS_PROJECT")

if not api_key:
    raise EnvironmentError(
        "HOPSWORKS_API_KEY environment variable not set. "
        "Add it to a local .env file or as a GitHub Actions secret."
    )

if not project_name:
    raise EnvironmentError(
        "HOPSWORKS_PROJECT environment variable not set. "
        "Add it to a local .env file or as a GitHub Actions secret."
    )

project = hopsworks.login(
    host="eu-west.cloud.hopsworks.ai",
    port=443,
    project=project_name,
    api_key_value=api_key,
)

print("Connected to Hopsworks!")
print("Project:", project.name)

# 2. GET FEATURE STORE + FEATURE GROUP

fs = project.get_feature_store()
print("Feature Store accessed successfully!")

print_section("FETCHING FEATURE GROUP")

feature_group = fs.get_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION,
)

print("Feature Group:", feature_group.name, "| Version:", feature_group.version)

# 3. BUILD QUERY 

print_section("BUILDING QUERY")

query = feature_group.select_except(EXCLUDED_COLUMNS)

print("Excluded columns:", EXCLUDED_COLUMNS)
print("Label columns   :", LABEL_COLUMNS)

# 4. CREATE / GET FEATURE VIEW

print_section("CREATING FEATURE VIEW")

feature_view = fs.get_or_create_feature_view(
    name=FEATURE_VIEW_NAME,
    version=FEATURE_VIEW_VERSION,
    query=query,
    labels=LABEL_COLUMNS,
    description=(
        "Karachi AQI feature view (V4 schema, forecast-lead features "
        "included). Labels: target_aqi_24/48/72. Excludes "
        "target_change_* (kept in the Feature Group only)."
    ),
)

print("Feature View:", feature_view.name, "| Version:", feature_view.version)

# 5. FINAL SUMMARY

print_section("FEATURE VIEW SETUP COMPLETED")

print("Feature Group:", feature_group.name, "| Version:", feature_group.version)
print("Feature View :", feature_view.name, "| Version:", feature_view.version)
print("Labels       :", LABEL_COLUMNS)
print("Excluded     :", EXCLUDED_COLUMNS)
print("\nReady for training-dataset creation or batch inference.")
print("=" * 60)