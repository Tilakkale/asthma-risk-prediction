# App Preview

This project includes a Streamlit clinical decision support interface for asthma risk assessment.

## Access status

There is currently **no deployed public Streamlit URL** in this repository. The addresses below work only while the app is running on the developer's machine or local network:

- Local: http://localhost:8501
- Network: http://<your-machine-ip>:8501

`http://<your-machine-ip>:8501` is not a public link. It is reachable only by devices permitted on the same network and while the local process is running.

## Publish a public link

To make the app available without `localhost`, push the repository to GitHub and deploy it with Streamlit Community Cloud:

1. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub.
2. Select **Create app**, then choose the repository, branch, and `app.py` entry point.
3. Deploy and copy the generated `https://<subdomain>.streamlit.app` address.

Do not publish a patient-facing app until the model has been retrained and independently validated. The current model fails its validation quality gate; its 100% recall operating point flags almost every assessment and has only 5.25% precision.

## Screenshot / demo notes

Add screenshots or a short demo video in this folder:

- docs/screenshots/

Suggested naming:

- app-dashboard.png
- risk-assessment-demo.mp4

## Run command

```powershell
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
