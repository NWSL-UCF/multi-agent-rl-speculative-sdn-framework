import csv
import fcntl
import logging
import os
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request
from collections import defaultdict
from datetime import timedelta
import pytz
import argparse
import pandas as pd

app = Flask(__name__)

# -------------------------- CONFIG --------------------------
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CSV_FILE = ""
LOG_FILENAME = "dashboard.log"
EXP_ID = "sim100"
def createExpBaseDirectory(args):
    os.makedirs(os.path.join(BASE_DIR, args.expId), exist_ok=True)

def setup_log(args):
    LOG_FILE = os.path.join(BASE_DIR, args.expId, LOG_FILENAME)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

STATUS_NOT_STARTED = "NOT_STARTED"
STATUS_SERVED = "SERVED"
STATUS_DONE = "DONE"
STATUS_ABORTED = "ABORTED"

# --------------------- HELPER FUNCTIONS -----------------------
def load_jobs():
    """Load jobs from the CSV file using file locking to prevent race conditions."""
    jobs = []
    try:
        with open(CSV_FILE, "r") as file:
            fcntl.flock(file, fcntl.LOCK_SH)  # Shared lock
            df = pd.read_csv(file)
            fcntl.flock(file, fcntl.LOCK_UN)  # Unlock

        # Convert field types
        df["id"] = df["id"].astype(int)
        df["request_timestamp"] = df["request_timestamp"].astype(float)
        df["completion_timestamp"] = df["completion_timestamp"].astype(float)
        df["required_time"] = df["required_time"].astype(float)

        # Extract machine name
        df["requested_by"] = df["requested_by"].fillna("Not Assigned")
        df["machine"] = df["requested_by"].apply(
            lambda x: "Not Assigned" if not isinstance(x, str) or x.strip() == "" else x.split("_")[0]
        )

        jobs = df.to_dict(orient="records")

    except Exception as e:
        logging.error(f"Error loading jobs from CSV: {e}")

    return jobs

def format_timestamp(timestamp):
    """Convert a Unix timestamp to human-readable format (EST) or return 'N/A' if invalid."""
    if timestamp < 0:
        return "N/A"
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S') if timestamp else "N/A"

def format_time(seconds):
    """Convert minutes to hh:mm:ss format."""
    return str(timedelta(seconds=round(seconds)))

def calculate_machine_stats(jobs):
    """Calculate statistics for each machine group."""
    machine_stats = defaultdict(lambda: {"count": 0, "total_time": 0, "instances": set()})
    total_completed = len([job for job in jobs if job["status"] == STATUS_DONE])
    
    for job in jobs:
        if job["status"] == STATUS_DONE:
            machine_name = job["requested_by"].split("_")[0]  # Extract machine prefix
            machine_stats[machine_name]["count"] += 1
            machine_stats[machine_name]["total_time"] += job["required_time"]
            machine_stats[machine_name]["instances"].add(job["requested_by"])
    
    for machine, data in machine_stats.items():
        data["average_time"] = format_time((data["total_time"] / data["count"]) if data["count"] else 0)
        data["percentage"] = (data["count"] / total_completed * 100) if total_completed else 0
        data["percentage"] = round(data["percentage"], 2)
        data["instance_count"] = len(data["instances"])
    
    return machine_stats

# ------------------------- JOB STATISTICS --------------------- 
@app.route("/job_stats", methods=["GET"])
def job_stats():
    jobs = load_jobs()
    interval = request.args.get("interval", "hourly")
    machine = request.args.get("machine", "all")
    edt = pytz.timezone("America/New_York")
    now = datetime.now(pytz.utc).astimezone(edt).timestamp()
    job_counts = defaultdict(int)
    total_jobs_completed = 0

    filtered_jobs = [job for job in jobs if job["status"] == STATUS_DONE and (machine == "all" or job["machine"] == machine)]
    
    if interval == "minutely":
        start_time = now - 1800
        x_labels = [(datetime.fromtimestamp(start_time + i * 60, edt)).strftime("%H:%M") for i in range(30)]
        for job in filtered_jobs:
            if job["completion_timestamp"] >= start_time:
                minute = int((job["completion_timestamp"] - start_time) // 60)
                job_counts[minute] += 1
                total_jobs_completed += 1
    elif interval == "hourly":
        start_time = now - 86400
        x_labels = [(datetime.fromtimestamp(start_time + i * 3600, edt)).strftime("%H:00") for i in range(24)]
        for job in filtered_jobs:
            if job["completion_timestamp"] >= start_time:
                hour = int((job["completion_timestamp"] - start_time) // 3600)
                job_counts[hour] += 1
                total_jobs_completed += 1
    else:
        first_day = min(job["completion_timestamp"] for job in filtered_jobs)
        days_elapsed = int((now - first_day) // 86400 + 1)
        x_labels = [(datetime.fromtimestamp(first_day + i * 86400, edt)).strftime("%b %d") for i in range(days_elapsed)]
        for job in filtered_jobs:
            day = int((job["completion_timestamp"] - first_day) // 86400)
            job_counts[day] += 1
            total_jobs_completed += 1

    y_values = [job_counts[i] for i in range(len(x_labels))]
    return jsonify({"labels": x_labels, "values": y_values, "total_jobs": total_jobs_completed})




# ------------------------ DASHBOARD ROUTE ---------------------
@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Display job statistics and job details in an HTML page with column-based sorting icons."""
    expId = EXP_ID
    jobs = load_jobs()
    total_jobs = len(jobs)
    machine_names = sorted(set(job["machine"] for job in jobs))
    total_jobs_served = sum(1 for job in jobs if job["status"] == STATUS_SERVED)
    total_jobs_completed = sum(1 for job in jobs if job["status"] == STATUS_DONE)
    total_jobs_aborted = sum(1 for job in jobs if job["status"] == STATUS_ABORTED)

    machine_stats = calculate_machine_stats(jobs)
        
    
    
    
    
    avg_completion_time = ""
    if total_jobs_completed > 0:
        total_time = sum(j["required_time"] for j in jobs if j["status"] == STATUS_DONE)
        avg_completion_time = format_time(total_time / total_jobs_completed)

    # ---------------------- HTML TEMPLATE ----------------------
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Job Dashboard</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: Arial, sans-serif; }
            .tab { display: flex; border-bottom: 2px solid #ddd; margin-bottom: 15px; }
            .tab button { background-color: #f1f1f1; border: none; padding: 10px 20px; cursor: pointer; margin-right: 5px; }
            .tab button.active { background-color: #ddd; font-weight: bold; }
            .tabcontent { display: none; padding: 10px; }
            .tabcontent.active { display: block; }
            table { width: 100%; border-collapse: collapse; margin-top: 10px; }
            th, td { border: 1px solid black; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .sort-icon {
                cursor: pointer;
                margin-left: 0.3em;
                font-size: 0.9em;
                color: #333;
                font-weight: bold; /* add this line */
            }

            /* Modal Styling */
            .modal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.4);
            }
            .modal-content {
                background-color: white;
                margin: 10% auto;
                padding: 20px;
                border: 1px solid #888;
                width: 50%;
                text-align: left;
                border-radius: 10px;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
                font-family: Arial, sans-serif;
            }
            .close {
                float: right;
                font-size: 28px;
                font-weight: bold;
                cursor: pointer;
            }

            /* Timeline Styling */
            .timeline {
                position: relative;
                padding-left: 20px;
                border-left: 3px solid #007bff;
                margin-top: 15px;
            }
            .timeline-item {
                position: relative;
                padding: 10px 0;
            }
            .timeline-item::before {
                content: "●";
                color: #007bff;
                position: absolute;
                left: -13px;
                top: 10px;
                font-size: 14px;
            }
            .timestamp {
                color: #666;
                font-size: 12px;
                font-style: italic;
            }
            .message {
                font-size: 16px;
                font-weight: bold;
            }
            .stats-box {
                position: absolute;
                top: 205px;
                left: 20px;
                background-color: white;
                border-radius: 12px;
                padding: 15px;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
                font-family: Arial, sans-serif;
                width: 280px;
            }
            .stats-box p {
                margin: 5px 0;
                font-size: 14px;
                color: #333;
            }
            .stats-box select {
                width: 100%;
                padding: 5px;
                border-radius: 5px;
                border: 1px solid #ccc;
                margin-top: 5px;
            }

        </style>
        <style>
            body { font-family: Arial, sans-serif; }
            .modal-2 {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.4);
            }
            .modal-content-2 {
                background-color: white;
                margin: 10% auto;
                padding: 20px;
                border-radius: 10px;
                width: 60%;
                text-align: center;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
            }
            .close {
                float: right;
                font-size: 28px;
                font-weight: bold;
                cursor: pointer;
            }
            .stats-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 16px;
            }
            .stats-table th, .stats-table td {
                padding: 10px;
                text-align: center;
                border-bottom: 1px solid #000000;
            }
            .stats-table th {
                background-color: #f2f2f2;
                color: black;
                font-weight: bold;
            }
            .modal-button {
                position: absolute;
                top: 123px;
                right: 1174px;
                padding: 12px 24px;
                font-size: 16px;
                cursor: pointer;
                background-color: #f2f2f2;
                color: black;
                border: none;
                border-radius: 5px;
                width: 260px;
            }
            #chart-container {
                width: 75%;
                height: 365px;
                margin: 20px 2px 20px 342px;
                justify-content: center;
                align-items: center;
                background: white;
                border-radius: 8px;
                padding: 10px;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            }
            canvas {
                max-width: 100%;
                max-height: 100%;
            }

            .total-jobs-box {
                position: relative;
                bottom: 10px;
                left: 10px;
                background-color: white;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 14px;
                font-weight: bold;
                color: #333;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                display: inline-block;
            }
            
            
            
            #jobChart {
                width: 100% !important;
                height: 100% !important;
                max-width: 1095px;
                max-height: 365px;
            }

        </style>
        <script>
            function openModal() {
                document.getElementById("statsModal").style.display = "block";
            }
            function closeModal() {
                document.getElementById("statsModal").style.display = "none";
            }
        </script>
        <script>
            function openTab(tabName) {
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                tablinks = document.getElementsByClassName("tablink");
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].classList.remove("active");
                }
                document.getElementById(tabName).style.display = "block";
                event.currentTarget.classList.add("active");
            }

            function showMessageModal(jobId, message) {
                const modal = document.getElementById("messageModal");
                const jobIdSpan = document.getElementById("jobId");
                const messageTimeline = document.getElementById("messageTimeline");

                // Set Job ID
                jobIdSpan.textContent = jobId;

                // Clear previous messages
                messageTimeline.innerHTML = "";

                try {
                    // Reverse the array so newest entries are first
                    const parsedMessage = JSON.parse(JSON.stringify(message)).reverse();

                    parsedMessage.forEach(entry => {
                    const item = document.createElement("div");
                    item.classList.add("timeline-item");

                    // Use a template literal (backticks) so ${...} is replaced at runtime
                    const timestamp = new Date(entry.timestamp * 1000).toLocaleString("en-US", {
                        weekday: "long",
                        year: "numeric",
                        month: "short",
                        day: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit",
                        hour12: true
                    });

                    // Template literal:
                    item.innerHTML = `
                        <div class="message">${entry.reason}</div>
                        <div class="timestamp">${timestamp}</div>
                    `;

                    messageTimeline.appendChild(item);
                    });
                } catch (e) {
                    messageTimeline.innerHTML = "<p>Invalid message format</p>";
                }

                modal.style.display = "block";
            }

            function closeMessageModal() {
                document.getElementById("messageModal").style.display = "none";
            }

            window.onclick = function(event) {
                const modal = document.getElementById("messageModal");
                if (event.target === modal) {
                    modal.style.display = "none";
                }
            };

            /**
            * Sort an HTML table (tbody rows) by a specific column index.
            * tableId: the ID of the <table> element
            * colIndex: the zero-based index of the column to sort
            * dataType: "string" | "number" for how values are compared
            * iconRef: The <span> element that was clicked (to track asc/desc)
            */
            function sortTable(tableId, colIndex, dataType, iconRef) {
                const table = document.getElementById(tableId);
                const tbody = table.querySelector("tbody");
                const rows = Array.from(tbody.querySelectorAll("tr"));

                // Check or toggle ascending/descending state on the icon
                let ascending = iconRef.getAttribute("data-ascending") === "true" ? false : true;
                iconRef.setAttribute("data-ascending", ascending ? "true" : "false");

                rows.sort((rowA, rowB) => {
                    const cellA = rowA.children[colIndex].innerText.trim();
                    const cellB = rowB.children[colIndex].innerText.trim();

                    // For numeric columns
                    if (dataType === "number") {
                        const valA = parseFloat(cellA) || 0;
                        const valB = parseFloat(cellB) || 0;
                        return ascending ? valA - valB : valB - valA;
                    } 
                    // Otherwise string compare
                    if (ascending) {
                        return cellA.localeCompare(cellB);
                    } else {
                        return cellB.localeCompare(cellA);
                    }
                });

                // Re-insert sorted rows into the table
                rows.forEach(row => tbody.appendChild(row));
            }
        </script>
    </head>
    <body>
        <h1 style="text-align: center; font-size: 32px; font-weight: bold; color: #333; background-color: #ddd; padding: 20px; border-radius: 8px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);">📊 Job Distribution Dashboard 📊</h1>
        <button class="modal-button" onclick="openModal()"> 
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-bar-chart" viewBox="0 0 16 16">
                <path d="M0 0h1v15h15v1H0V0zM6 12V5h1v7H6zm4 0V3h1v9h-1zm4 0V7h1v5h-1z"/>
            </svg>
            Show Machine Statistics
        </button>
        <div id="statsModal" class="modal-2">
            <div class="modal-content-2">
                <span class="close" onclick="closeModal()">&times;</span>
                <h2>Machine Specific Stats</h2>
                <table class="stats-table">
                    <tr>
                        <th>Name</th>
                        <th>Instances</th>
                        <th>Jobs Done</th>
                        <th>Avg Completion Time</th>
                        <th>Percentage of Share</th>
                    </tr>
                    {% for machine, data in machine_stats.items() %}
                    <tr>
                        <td>{{ machine }}</td>
                        <td>{{ data.instance_count }}</td>
                        <td>{{ data.count }}</td>
                        <td>{{ data.average_time }}</td>
                        <td>{{ data.percentage }}%</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </div>
        <div class="stats-box">
            <p><strong>Experiment ID:</strong> {{ expId }}</p>
            <p><strong>Total Jobs:</strong> {{ total_jobs }}</p>
            <p><strong>Total Jobs Running (SERVED):</strong> {{ total_jobs_served }}</p>
            <p><strong>Total Jobs Completed:</strong> {{ total_jobs_completed }}</p>
            <p><strong>Average Completion Time:</strong> {{ avg_completion_time }}</p>
            <p><strong>Total Jobs Aborted:</strong> {{ total_jobs_aborted }}</p>
            <p><strong>Total Jobs for Selected Interval:</strong> <span id="totalJobs"></span></p>
            <strong><label for="machineFilter">Select Machine:</label></strong>
            <select id="machineFilter" onchange="updateChart()">
                <option value="all">All Machines</option>
                {% for machine in machine_names %}
                    <option value="{{ machine }}">{{ machine }}</option>
                {% endfor %}
            </select> 
            <strong><label for="timeInterval">Select Interval:</label></strong>
            <select id="timeInterval" onchange="updateChart()">
                <option value="hourly">Hourly</option>
                <option value="minutely">Minutely</option>
                <option value="daily">Daily</option>
            </select>
        </div>
        <div id="chart-container">
            <canvas id="jobChart"></canvas>
        </div>
        <script>
            let chart;
            function updateChart() {
                let interval = document.getElementById("timeInterval").value;
                let machine = document.getElementById("machineFilter").value;
                fetch(`/job_stats?interval=` + interval + `&machine=` + machine)
                    .then(response => response.json())
                    .then(data => {
                        if (chart) { chart.destroy(); }
                        let ctx = document.getElementById("jobChart").getContext("2d");
                        chart = new Chart(ctx, {
                            type: "bar",
                            data: {
                                labels: data.labels,
                                datasets: [{
                                    label: "Jobs Completed",
                                    data: data.values,
                                    backgroundColor: "rgba(54, 162, 235, 0.6)",
                                    borderColor: "rgba(54, 162, 235, 1)",
                                    borderWidth: 1
                                }]
                            },
                            options: {
                                scales: { y: { beginAtZero: true } }
                            }
                        });
                        // Update the total jobs count in the HTML element with id "totalJobs"
                        document.getElementById("totalJobs").innerText = data.total_jobs;
                    });
            }
            updateChart();
        </script>
        
        <!-- Tab Buttons -->
        <div class="tab">
            <button class="tablink active" onclick="openTab('SERVED')">SERVED</button>
            <button class="tablink" onclick="openTab('DONE')">DONE</button>
            <button class="tablink" onclick="openTab('ABORTED')">ABORTED</button>
            <button class="tablink" onclick="openTab('NOT_STARTED')">NOT_STARTED</button>
        </div>

        <!-- Modal -->
        <div id="messageModal" class="modal">
            <div class="modal-content">
                <span class="close" onclick="closeMessageModal()">&times;</span>
                <h2>Job History</h2>
                <p><strong>Job ID:</strong> <span id="jobId"></span></p>
                <div class="timeline" id="messageTimeline"></div>
            </div>
        </div>

        {% for status in ['SERVED', 'DONE', 'ABORTED', 'NOT_STARTED'] %}
            <div id="{{ status }}" class="tabcontent {% if loop.first %}active{% endif %}">
                <h2>{{ status }} Jobs</h2>
                <!-- Unique table ID per status -->
                <table id="{{ status|lower }}-table">
                    <thead>
                        <tr>
                            <!-- Example: Only the icon is clickable (the <span>) -->
                            <th>
                                ID
                                <span class="sort-icon"
                                      data-ascending="false"
                                      style="font-weight: bold;"
                                      onclick="sortTable('{{ status|lower }}-table', 0, 'number', this)">
                                    &#8597;
                                </span>
                            </th>
                            <th>
                                Requested By
                                <span class="sort-icon"
                                      data-ascending="false"
                                      style="font-weight: bold;"
                                      onclick="sortTable('{{ status|lower }}-table', 1, 'string', this)">
                                    &#8597;
                                </span>
                            </th>
                            <th>
                                Status
                                <span class="sort-icon"
                                      data-ascending="false"
                                      style="font-weight: bold;"
                                      onclick="sortTable('{{ status|lower }}-table', 2, 'string', this)">
                                    &#8597;
                                </span>
                            </th>
                            <th>
                                Request Time (EDT)
                                <span class="sort-icon"
                                      data-ascending="false"
                                      style="font-weight: bold;"
                                      onclick="sortTable('{{ status|lower }}-table', 3, 'string', this)">
                                    &#8597;
                                </span>
                            </th>
                            <th>
                                Completion Time (EDT)
                                <span class="sort-icon"
                                      data-ascending="false"
                                      style="font-weight: bold;"
                                      onclick="sortTable('{{ status|lower }}-table', 4, 'string', this)">
                                    &#8597;
                                </span>
                            </th>
                            <th>
                                Required Time (Min)
                                <span class="sort-icon"
                                      data-ascending="false"
                                      style="font-weight: bold;"
                                      onclick="sortTable('{{ status|lower }}-table', 5, 'number', this)">
                                    &#8597;
                                </span>
                            </th>
                            <th>Message</th>
                        </tr>
                    </thead>
                    <tbody>
                    {% for job in jobs if job.status == status %}
                        <tr>
                            <td>{{ job.id }}</td>
                            <td>{{ job.requested_by }}</td>
                            <td>{{ job.status }}</td>
                            <td>{{ format_timestamp(job.request_timestamp - (60*60*4)) }}</td>
                            <td>{{ format_timestamp(job.completion_timestamp - (60*60*4)) }}</td>
                            <td>{{ (job.required_time / 60) | round }}</td>
                            <td>
                                <button onclick="showMessageModal({{ job.id }}, {{ job.message }})">View Message</button>
                            </td>
                        </tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
        {% endfor %}
    </body>
    </html>
    """

    return render_template_string(
        html_template,
        expId = expId,
        total_jobs=total_jobs,
        total_jobs_served=total_jobs_served,
        total_jobs_completed=total_jobs_completed,
        total_jobs_aborted=total_jobs_aborted,
        avg_completion_time=avg_completion_time,
        jobs=jobs,
        format_timestamp=format_timestamp,
        machine_stats=machine_stats,
        machine_names=machine_names
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Flask Dashboard server")
    parser.add_argument("--host", default="0.0.0.0", help="IP address to bind to")
    parser.add_argument("--jobDB", default="jobs.csv", help="CSV file (<filename>.csv) placed in the same directory as server.py")
    parser.add_argument("--port", type=int, default=5050, help="Port number to listen on")
    parser.add_argument("--expId", type=str, default="sim1", help="Give an unique name")
    args = parser.parse_args()
    createExpBaseDirectory(args)
    setup_log(args)
    logging.info(f"Starting Flask Dashboard server on {args.host}:{args.port}...")
    CSV_FILE = os.path.join(BASE_DIR, args.expId, args.jobDB)
    EXP_ID = args.expId
    app.run(host=args.host, port=args.port)  

# python dashboard.py --expId=sim1 --jobDB=jobs.csv --host=0.0.0.0 --timeoutLimit=5050
