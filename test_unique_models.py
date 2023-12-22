from fastapi.testclient import TestClient
import app 

def test_get_models():
    client = TestClient(app.app)
    response = client.get("/models")
    assert response.status_code == 200
    assert response.json() == {"models": ["LGBM", "GBClassifier"]}