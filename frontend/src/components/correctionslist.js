'use strict';
import React, {Component} from 'react';

class CorrectionsList extends Component {
    constructor(props) {
        super(props);
        this.state = {};
    }

    componentDidMount() {
        let count = 0;
        let word_array = Object.keys(this.props.corrections).sort();
        this.setState({word_array: word_array})
        for (const word of word_array) {
            if (count === 0) {
                this.setState({[word + '_show']: true});

            } else {
                this.setState({[word + '_show']: false});
            }
            this.setState({[word + '_submitted']: false});
            count++;
        }
        this.setState({'showSpelling': true})

    }

    unorderedList = function () {
        let word_array = this.state.word_array;
        return (
            word_array.map((word) =>

                this.state[word + '_show'] ?
                    <div key={word}>
                        <span className="display-4">By {word}, did you mean one of the following?</span>
                        <ul className="list-group display-2">
                            {this.listItems(word, this.props.corrections[word])}
                            <li key="none_of_the_above">
                                <button type="button" className="btn btn-secondary btn-lg btn-block"
                                        onClick={(event) => this.handleClick(event, word, null)}>None of the
                                    above
                                </button>
                            </li>
                        </ul>
                    </div>
                    : null)
        )
    };

    listItems = function (original_word, corrections) {
        return (
            corrections.map((correction, index) =>
                <li key={correction.toString()}>
                    <button type="button" className="btn btn-primary btn-lg btn-block"
                            onClick={(event) => this.handleClick(event, original_word, correction)}>{correction}</button>
                </li>
            )
        )
    };

    handleClick = (event, original_word, correction) => {
        this.hideCorrections(original_word);
        if (correction) {
            this.props.sendSpellingCorrection(this.props.response_id, original_word, correction);
        }
        let next_index = this.state.word_array.indexOf(original_word) + 1;
        if (next_index < this.state.word_array.length) {
            let next_word = this.state.word_array[next_index];
            this.showCorrections(next_word);
        } else {
            // reload the overall component and obtain a new cue
            this.props.reload()
        }


    };


    hideCorrections = (word) => {
        this.setState({[word + '_show']: false});
        this.setState({[word + '_submitted']: true});
    };

    showCorrections = (word) => {
        this.setState({[word + '_show']: true});
        this.setState({[word + '_submitted']: false});
    };

    render() {
        //make sure the state has the first key at least
        return this.state.showSpelling ? (
            <div>
                {this.unorderedList()}
            </div>
        ) : null
    }
}


export default CorrectionsList;
