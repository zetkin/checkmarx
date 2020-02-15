import React from 'react';
import ImageUploader from 'react-images-upload';
import logo from './img/logo.png';
import './App.css';

class App extends React.Component {

    constructor(props) {
      super(props);
      this.state = {
        picture: null,
      };
    }
    
    render() {
      return (
        <div className="App">
          <header className="App-header">
            <img src={logo} className="App-logo" alt="logo" />
          </header>
          <ImageUploader withIcon
            buttonText='Choose Image'
            onChange={(picture) => { this.setState({ picture }); }}
            imgExtension={['.jpg', '.png']}
          />
          <button
            onClick={()=>{}} disabled={!this.state.picture}>Check Marx</button>
        </div>
      );
    }
}

export default App;
