# Set the date format
DATE=$(date +%Y_%m_%d)
REPORT_DIR="results/daily_reports"
TEMPLATE="results/daily_reports/daily_report_template.md"
NEW_REPORT="${REPORT_DIR}/report_${DATE}.md"

# Check if the report already exists
if [ -e "${NEW_REPORT}" ]; then
    echo "Report for ${DATE} already exists."
else
    # Copy the template to create a new report
    cp "${TEMPLATE}" "${NEW_REPORT}"
    echo "Created new report: ${NEW_REPORT}"
fi