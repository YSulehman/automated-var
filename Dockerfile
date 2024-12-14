# specify base image
FROM python:3.10-slim

# specify working directory
WORKDIR /automated-var

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy over the rest of the files
COPY . .

# set the default command to execute
CMD ["python", "assistant_referee.py"]