# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install any dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose the port that your Flask app runs on
EXPOSE 6000

# Set environment variables for Flask
ENV FLASK_APP=RLActionPlanner.py
ENV FLASK_ENV=production

# Command to run the Flask app
CMD ["python", "app.py"]