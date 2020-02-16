import React from 'react';
import ImageUploader from 'react-images-upload';
import axios from 'axios';
import logo from './img/logo.png';
import Webcam from "react-webcam";
import './App.css';

function b64toBlob(byteCharacters, contentType, sliceSize) {
        contentType = contentType || '';
        sliceSize = sliceSize || 512;

        byteCharacters = byteCharacters.slice("data:image/jpeg;base64,".length)
        var byteCharacters = atob(byteCharacters);
        var byteArrays = [];

        for (var offset = 0; offset < byteCharacters.length; offset += sliceSize) {
            var slice = byteCharacters.slice(offset, offset + sliceSize);

            var byteNumbers = new Array(slice.length);
            for (var i = 0; i < slice.length; i++) {
                byteNumbers[i] = slice.charCodeAt(i);
            }

            var byteArray = new Uint8Array(byteNumbers);

            byteArrays.push(byteArray);
        }

      var blob = new Blob(byteArrays, {type: contentType});
      return blob;
}

class App extends React.Component {

    constructor(props) {
      super(props);
      this.state = {
        picture: null,
        result: [],
      };
      this.webcamRef = React.createRef();
    }

    render() {
      const videoConstraints = {
        width: 1280,
        height: 720,
        facingMode: "environment"
      };
      const capture = () => {
        const imageSrc = this.webcamRef.current.getScreenshot();
        this.setState({picture: imageSrc});
      };
      const paragraphs = [];
      for (var i = 0; i < this.state.result.length; i++) {
          paragraphs.push(<p key={this.state.result[i]}>{this.state.result[i]} was Marxed</p>);
      }
      const image = this.state.picture ?
        <img src={this.state.picture} alt="webcam capture" /> : <Webcam
        audio={false}
        height={720}
        ref={this.webcamRef}
        screenshotFormat="image/jpeg"
        width={1280}
        videoConstraints={videoConstraints}
      />

      return (
        <div className="App">
          <header className="App-header">
            <img src={logo} className="App-logo" alt="logo" />
          </header>
          {image}
          <button onClick={capture}>Capture photo</button>
          <button
            onClick={()=>{
                var form = new FormData();
                form.append("image", b64toBlob(this.state.picture));
                axios({
                    method: "post",
                    url: "/scan",
                    data: form,
                    headers: { 'Content-Type': 'multipart/form-data' }
                })
                .then((res) => {
                    this.setState({ picture: null, result: res.data.result });
                })
                .catch((err) => {
                    console.error(err);
                });
            }} disabled={!this.state.picture}>Check Marx
          </button>
          {paragraphs}
        </div>
      );
    }
}

export default App;
