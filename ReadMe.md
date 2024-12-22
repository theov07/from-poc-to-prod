# Run the application
```bash 
python app.py
```
## CURL request 
```bash 
curl -X POST -H "Content-Type: application/json" -d '{"text": "help me with my javascript issue", "top_k":5}' http://localhost:5000/predict
```
## Results

```string 
The text is :  help me with my javascript issue  and the prediction is :  ['javascript', 'iphone', 'java', 'c#', 'php']  
```