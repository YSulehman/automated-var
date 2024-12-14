# specify base image
FROM 

# specify working directory
WORKDIR

# install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# set the default command to execute
CMD ["python", "assistant_referee.py"]