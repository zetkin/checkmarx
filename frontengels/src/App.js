import React from 'react';
import ImageUploader from 'react-images-upload';
import axios from 'axios';
import logo from './img/logo.png';
import './App.css';

class App extends React.Component {

    constructor(props) {
      super(props);
      this.state = {
        picture: null,
        result: [],
      };
    }

    render() {
      const paragraphs = [];
      for (var i = 0; i < this.state.result.length; i++) {
          paragraphs.push(<p key={this.state.result[i]}>{this.state.result[i]} was Marxed</p>);
      }
      return (
        <div className="App">
          <header className="App-header">
            <img src={logo} className="App-logo" alt="logo" />
          </header>
          <ImageUploader withIcon
            buttonText='Choose Image'
            onChange={(picture) => { this.setState({ picture }); }}
            imgExtension={['.jpg', '.png']}
            singleImage
          />
          <button
            onClick={()=>{
                console.log(this.state.picture[0]);
                var form = new FormData();
                form.append("image", this.state.picture[0]);
                axios({
                    method: "post",
                    url: "/scan",
                    data: form,
                    headers: { 'Content-Type': 'multipart/form-data' }
                })
                .then((res) => {
                    console.log(res);
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
