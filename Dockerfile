
        FROM python:3.9

        WORKDIR /app

        # Copy requirements and install dependencies
        COPY requirements.txt requirements.txt
        RUN pip install -r requirements.txt

        # Copy all project files
        COPY . .

        # Expose the port the app runs on
        EXPOSE 5000

        # Command to run the application
        CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "ml2:app"]
        