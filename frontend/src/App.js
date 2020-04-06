import React, {Component} from 'react';
import logo from './logo.svg';
import './App.css';
import InputMask from 'react-input-mask';
import Spelling from './components/spelling.js'
import Fingerprint2 from 'fingerprintjs2'
import ls from 'local-storage'
import * as api from "./helpers/api.js"

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
        this.state = {response: '', cue: '', time: 0, corrections: null};
    }

    componentDidMount() {
        api.getCueWord().then(cue_obj => {
            this.setState({
                cue: cue_obj['cue'],
                response: '',
                time_start: Date.now()
            });
            this.getFingerprint();
        }).catch(err => console.log('There was an error:' + err));
    }

    mySubmitHandler = (event) => {
        event.preventDefault();
        let response = this.state.response;
        let cue = this.state.cue;
        let time_duration = Date.now() - this.state.time_start;
        let user_id = ls.get('user_id')
        // TODO implement spelling options
        api.saveResponseWord(cue, response, time_duration, user_id).then((res) => {
            if (res.status === 201) {
                return res.json();
            }
        }).then((data) => {
            if (data.hasOwnProperty('corrections')) {
                this.spellingOptions(data);
            } else {
                this.reload()
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
            let callbackId = requestIdleCallback(this.idleFingerprint());
        } else {
            setTimeout(this.idleFingerprint.bind(this), 500)
        }
    }

    idleFingerprint() {
        Fingerprint2.get(function (components, result) {
            //console.log(components); // an array of components: {key: ..., value: ...}
            var values = components.map(function (component) {
                return component.value
            });
            var murmur = Fingerprint2.x64hash128(values.join(''), 31)
            ls.set('user_id', murmur)
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
