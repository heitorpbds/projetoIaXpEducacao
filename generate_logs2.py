import datetime
import random
import os

def generate_apache_like_log(filename="apache_like_log.log", num_lines=1000000):
    start_date = datetime.datetime(2024, 1, 1, 0, 0, 0)
    
    log_levels = ["INFO", "ERROR", "DEBUG", "WARN"]
    
    info_messages = [
        "User {} logged in",
        "User {} accessed dashboard",
        "Resource {} loaded successfully",
        "Application started",
        "Service {} initialized",
        "Configuration reloaded",
        "Session {} created",
    ]
    
    error_messages = [
        "Failed to connect to database",
        "Authentication failed for user {}",
        "Resource {} not found",
        "Internal server error",
        "Disk space low",
        "Memory allocation failed",
        "Invalid request from {}",
    ]
    
    debug_messages = [
        "Query executed in {}ms",
        "Function {} called with args {}",
        "Variable {} has value {}",
        "Processing request from {}",
        "Database transaction started",
        "Cache hit for key {}",
    ]
    
    warn_messages = [
        "Connection pool running low",
        "Deprecated API used by {}",
        "Slow query detected ({}ms)",
        "Unauthorized access attempt from {}",
        "High CPU usage ({}%)",
        "Too many open files",
    ]

    print(f"Gerando arquivo de log '{filename}' com {num_lines} linhas...")

    with open(filename, 'w') as f:
        for i in range(num_lines):
            current_date = start_date + datetime.timedelta(seconds=random.randint(0, 31536000)) # Adiciona até 1 ano de segundos
            timestamp = current_date.strftime("%Y-%m-%d %H:%M:%S")
            
            log_level = random.choice(log_levels)
            
            message = ""
            if log_level == "INFO":
                msg_template = random.choice(info_messages)
                if "{}" in msg_template:
                    if "User" in msg_template:
                        message = msg_template.format(random.choice(["admin", "guest", "user" + str(random.randint(1, 100))]))
                    elif "Resource" in msg_template:
                        message = msg_template.format(random.choice(["/api/data", "/home", "/reports"]))
                    elif "Service" in msg_template:
                        message = msg_template.format(random.choice(["auth_service", "data_processor"]))
                    elif "Session" in msg_template:
                        message = msg_template.format(random.randint(10000, 99999))
                else:
                    message = msg_template
            elif log_level == "ERROR":
                msg_template = random.choice(error_messages)
                if "{}" in msg_template:
                    if "user" in msg_template:
                        message = msg_template.format(random.choice(["admin", "guest", "user" + str(random.randint(1, 100))]))
                    elif "Resource" in msg_template:
                        message = msg_template.format(random.choice(["/api/users", "/config.json"]))
                    elif "request from" in msg_template:
                        message = msg_template.format("192.168.1." + str(random.randint(1, 254)))
                else:
                    message = msg_template
            elif log_level == "DEBUG":
                msg_template = random.choice(debug_messages)
                if "{}" in msg_template:
                    if "Query executed" in msg_template:
                        message = msg_template.format(random.randint(10, 500))
                    elif "Function" in msg_template:
                        message = msg_template.format(random.choice(["getUserData", "processOrder"]), f"({random.randint(1,5)}, 'test')")
                    elif "Variable" in msg_template:
                        message = msg_template.format(random.choice(["id", "status"]), random.choice([123, "active", True]))
                    elif "request from" in msg_template:
                        message = msg_template.format("10.0.0." + str(random.randint(1, 254)))
                    elif "key" in msg_template:
                        message = msg_template.format("item_" + str(random.randint(1, 1000)))
                else:
                    message = msg_template
            elif log_level == "WARN":
                msg_template = random.choice(warn_messages)
                if "{}" in msg_template:
                    if "deprecated API" in msg_template:
                        message = msg_template.format(random.choice(["module_x", "old_function"]))
                    elif "Slow query" in msg_template:
                        message = msg_template.format(random.randint(500, 5000))
                    elif "access attempt" in msg_template:
                        message = msg_template.format("203.0.113." + str(random.randint(1, 254)))
                    elif "CPU usage" in msg_template:
                        message = msg_template.format(random.randint(70, 99))
                else:
                    message = msg_template
            
            f.write(f"{timestamp} {log_level} {message}\n")
            
            if (i + 1) % 100000 == 0:
                print(f"  {i + 1} linhas geradas...")

    print(f"Geração concluída. O arquivo '{filename}' foi criado com {num_lines} linhas.")
    print(f"Tamanho aproximado do arquivo: {os.path.getsize(filename) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    generate_apache_like_log()