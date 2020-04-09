import React, {Component} from 'react';
import './App.css';
import InputMask from 'react-input-mask';
import Fingerprint2 from 'fingerprintjs2'
import ls from 'local-storage'
import * as api from "./helpers/api.js"
import CorrectionsList from "./components/correctionslist";

class App extends Component {
    render() {
        return (
            <div className="App">
                <WordAssociation></WordAssociation>
            </div>
        );
    }
}

//TODO prevent form submission when field is blank
//TODO allow spaces in form submission


class WordAssociation extends Component {
    constructor(props) {
        super(props);
        this.state = {response: '', cue: '', time: 0, corrections: null, showSpelling: false};
        this.sendSpellingCorrection = this.sendSpellingCorrection.bind(this)
        this.reload = this.reload.bind(this);
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
        let user_id = ls.get('user_id');
        // TODO implement spelling options
        api.saveResponseWord(cue, response, time_duration, user_id).then((res) => {
            if (res.status === 201) {
                return res.json();
            }
        }).then((data) => {
            console.log(data);
            this.setState({'response_id': data['response_id']})
            if (data.hasOwnProperty('corrections')) {
                // corrections must be set before showing the spelling component
                this.setState({corrections: data['corrections']});
                this.setState({showSpelling: true});
            } else {
                this.reload()
            }
        });
    };

    myChangeHandler = (event) => {
        this.setState({response: event.target.value});
    };

    reload() {
        this.setState({showSpelling: false});
        this.componentDidMount();
    }

    getFingerprint() {
        if (window.requestIdleCallback) {
            requestIdleCallback(this.idleFingerprint.bind(this));
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
            var murmur = Fingerprint2.x64hash128(values.join(''), 31);
            ls.set('user_id', murmur)
        })
    }

    sendSpellingCorrection(response_id, original__word, correction) {
        api.correctSpelling(response_id, original__word, correction).then((res) => {
            if (res.status === 201) {
                console.log('spelling sent');
            }
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
                            <CorrectionsList
                                corrections={this.state.corrections}
                                response_id={this.state.response_id}
                                showSpelling={this.state.showSpelling}
                                sendSpellingCorrection={this.sendSpellingCorrection}
                                reload={this.reload}/> : null}
                    </div>
                </div>
            </form>
        );
    }
}


class ResponseInput extends React.Component {
    render() {
        let mask = /^[a-z\s]{1,24}$/;
        return <InputMask {...this.props} mask={mask} maskChar=""/>;
    }
}

export default App;
