import React, {Component} from 'react';
import logo from './logo.svg';
import './App.css';
import InputMask from 'react-input-mask';
import Spelling from './components/spelling.js'
import Fingerprint2 from 'fingerprintjs2'

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
        this.state = {response: '', cue: '', time: 0, spelling: null};
    }

    componentDidMount() {
        getCueWord().then(cue_obj => {
            this.setState({
                cue: cue_obj['cue'],
                response: '',
                time_start: Date.now()
            })
            this.getFingerprint();
        }).catch(err => console.log('There was an error:' + err));
    }

    mySubmitHandler = (event) => {
        event.preventDefault();
        let response = this.state.response;
        let cue = this.state.cue;
        let time_duration = Date.now() - this.state.time_start;
        // TODO implement spelling options
        saveResponseWord(cue, response, time_duration).then((res) => {
            if (res.status === 201) {
                this.reload()
            } else if (res.status === 406) {
                return res.json();
            }
        }).then((data) => {
            if (data) {
                this.spellingOptions(data);
            }
        });
    };

    myChangeHandler = (event) => {
        this.setState({response: event.target.value});
    };

    reload() {
        this.componentDidMount();
    }

    spellingOptions(corrections_obj) {
        console.log(corrections_obj);
        // we must set the corrections before showing the component
        this.setState({corrections: corrections_obj['corrections']});
        this.setState({showSpelling: true});

    }

    getFingerprint() {
        if (window.requestIdleCallback) {
            let callbackId = requestIdleCallback(this.idleFingerprint(this.setState.bind(this)));
        } else {
            setTimeout(function () {
                Fingerprint2.get(function (components) {
                    console.log(components); // an array of components: {key: ..., value: ...}

                })
            }.bind(this), 500)
        }
    }

    idleFingerprint(setState) {
        Fingerprint2.get(function (components) {
            //console.log(components); // an array of components: {key: ..., value: ...}
            console.log('inner');
            setState({'user_id': components})
        })
    }


    render() {
        return (
            <form onSubmit={this.mySubmitHandler} className="display-1">
                <div>
                    <div>
                        {!this.state.showSpelling ?
                            <div>
                                <span id="cue" className="m-2">{this.state.cue}</span>
                                <ResponseInput id="response" className="m-2" autoComplete="off"
                                               onChange={this.myChangeHandler} type="text"
                                               value={this.state.response}></ResponseInput>

                                <input type="submit"/>
                            </div>
                            : null}
                        {this.state.showSpelling ?
                            <SpellingCheck response={this.state.response} corrections={this.state.corrections}/> : null}
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


class SpellingCheck extends React.Component {
    render() {
        return <Spelling {...this.props}/>
    }
}


class ResponseInput extends React.Component {
    render() {
        return <InputMask {...this.props} mask="aaaaaaaaaaaaaaaaaaaaaaaa" maskChar=""/>;
    }
}

export default App;
