import React, {Component} from 'react';
import logo from './logo.svg';
import './App.css';
import InputMask from 'react-input-mask';

class App extends Component {
    render() {
        return (
            <div className="App">
                <WordAssociation></WordAssociation>
            </div>
        );
    }
}

class WordAssociation extends Component {
    constructor(props) {
        super(props);
        this.state = {response: '', cue: '', time: 0};
    }

    componentDidMount() {
        getCueWord().then(cue_obj => {
            this.setState({
                cue: cue_obj['cue'],
                response: '',
                time_start: Date.now()
            })
        }).catch(err => console.log('There was an error:' + err));
    }

    mySubmitHandler = (event) => {
        event.preventDefault();
        let response = this.state.response;
        let cue = this.state.cue;
        let time_duration = Date.now() - this.state.time_start;
        saveResponseWord(cue, response, time_duration).then(saveResponseResult => {
            if (saveResponseResult.status === 200) {
                this.reload();
            }
        });

    };

    myChangeHandler = (event) => {
        this.setState({response: event.target.value});
    };

    reload() {
        this.componentDidMount();
    }


    render() {
        return (
            <form onSubmit={this.mySubmitHandler} className="display-1">
                <div>
                    <div>
                        <span id="cue" className="m-2">{this.state.cue}</span>
                        <ResponseInput id="response" className="m-2" autoComplete="off"
                                       onChange={this.myChangeHandler} type="text"
                                       value={this.state.response}></ResponseInput>
                        <input type="submit"/>
                    </div>
                </div>
            </form>
        );
    }
}


function getCueWord() {
    return fetch(process.env.REACT_APP_API_URL + 'cueresponse',
        {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                "Accept": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        }).then((res) => {
        return res.json();
    }).then((data) => {
        return data;
    });
}

function saveResponseWord(cue, response, time_duration) {
    return fetch(process.env.REACT_APP_API_URL + "cueresponse",
        {
            method: 'POST',
            body: JSON.stringify({
                'cue': cue,
                'response': response,
                'time_duration': time_duration
            }),
            headers: {
                //'X-Api-Key': API_KEY,
                'Content-Type': 'application/json',
                "Accept": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        }
    ).then(function (response) {
        return response
    });
}


class ResponseInput extends React.Component {
    render() {
        return <InputMask {...this.props} mask="aaaaaaaaaaaaaaaaaaaaaaaa" maskChar=""/>;
    }
}

export default App;
